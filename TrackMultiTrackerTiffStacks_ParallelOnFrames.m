
function TrackMultiTrackerTiffStacks_ParallelOnFrames(nParalell, StabilizeTaps, TapIndStart, TapIndEnd)

% StabilizeTaps = true; % adjust for motion artifact in tapping experiments
% using 2d cross correlation and an erode on the background image. This is
% done for the first 150 frames, while the plate is shaking due to the
% strong solenoid tap
%
% TapIndStart = 270; 
% TapIndEnd = 299

% the indexes of where the taps happen, default
% between 270 and 299
%
% 

if nargin == 0
    StabilizeTaps = false;
    TapIndStart = 99999;
    TapIndEnd = 99999;
    nParalell = 4;
    disp('nPar = 4, Not stabilizing images for taps');
elseif nargin == 1
    if StabilizeTaps == 1
       
       TapIndStart = 270; % the indexes of where the taps happen
       TapIndEnd = 299;
       disp('Stabilizing between 270 and 299')
    else
       TapIndStart = 99999;
       TapIndEnd = 99999;
       disp('Not stabilizing images for taps');
    end
elseif nargin == 4
    disp(['Stabilizing images for taps between ', num2str(TapIndStart), ' and ', num2str(TapIndEnd)]);
else
    %error('wrong number of arguments');
end
%nParalell = 4; % number of paralell threads
SaveFishSkel = true; % saves a file with the fish skeleton, in addition to kinematic parameters
FishLen = 50; % the ~ length of a fish in pixels
BodyThresh = 15; % the threshold used to find the fish
HeadFiltSize = round(FishLen/15); % smoothing kernel to find heads
TailFiltSize = 3; % smaller smoothing kernel to find points along the tails
nSeg = 8; % number of segments to try and reconstruct along the tail
len =5; % length of each segment
SearchAngle = pi/3; % search angle for finding the next segment

% acceptable sizes of the fish
minFishSize = 50;
maxFishSize = 500;


f = fspecial('disk', HeadFiltSize);

BkgThresh = 4; % the minimum background subtracted intensity value that will be accepted for reconstructing the tail.


Trials = uipickfiles('Prompt', 'Pick the Tiff Stacks', 'FilterSpec', '*0.tif');
NumTrials = length(Trials);

[TiffDir, TiffName, ext] = fileparts(Trials{1});
cd(TiffDir);

% load the proper ROI file
if ~isempty(strfind(TiffName, 'plate_0_'));
    ROIfile = dlmread('ROI_defs_0.txt');
elseif ~isempty(strfind(TiffName, 'plate_1_'));
    ROIfile = dlmread('ROI_defs_1.txt');
else
    ROIfile = dlmread('CCD_ROIDefs_NoICCD.txt');
end

nROIs = size(ROIfile, 1);

% start the paralell pools, if not already started
CurrentPool = gcp('nocreate');
if sum(size(CurrentPool)) == 0;
    myCluster = parcluster;
    myCluster.NumWorkers = nParalell;
    parobj = parpool('local', nParalell);
end


for nTrial = 1:NumTrials
    
    tic
    
    % Get the TIFF info
    [TiffDir, TiffName, ext] = fileparts(Trials{nTrial});
    cd(TiffDir);
    TiffName = [TiffName, ext];
    TiffName2nd = strrep(TiffName, '00.tif', '01.tif');
    TiffName3rd = strrep(TiffName, '00.tif', '02.tif');
    TiffInfo = dir(TiffName);
    TiffDate = TiffInfo.date;
    
    TrialInd = str2num(TiffName(end-12:end-9));
    % Load ROI file
    if ~isempty(strfind(TiffName, 'plate_0_'));
        ROIfile = dlmread('ROI_defs_0.txt');
    elseif ~isempty(strfind(TiffName, 'plate_1_'));
        ROIfile = dlmread('ROI_defs_1.txt');
    else
        ROIfile = dlmread('CCD_ROIDefs_NoICCD.txt');
    end
    
    % reorder ROI file, MH ROIs have height and widht reversed:
    
    ROIfile(:, 3:4) = ROIfile(:, [4,3]);
    nROIs = size(ROIfile, 1);
    
    InfoImage=imfinfo(TiffName);
    Height = InfoImage(1).Height;
    Width = InfoImage(1).Width;
    %
    
    Frames = length(imfinfo(TiffName));
    try
        Frames_end = length(imfinfo(TiffName2nd));
        FramesTotal = Frames + Frames_end;
    catch
        warning(['didnt find stack: ', TiffName2nd]);
        FramesTotal = Frames;
    end
    
    Bkg = LoadTiffPlane8Bit(TiffDir, TiffName, 1); % the first frame of the tiff is the running background image
    Mid = LoadTiffPlane8Bit(TiffDir, TiffName, floor(Frames/2));
    Last = LoadTiffPlane8Bit(TiffDir, TiffName2nd, Frames_end);
    
    % take the max between the bkg image, and the ~ middle image, and the last image of the burst. This way fish that were still in the background image but then moved should be trackable
    
    Bkg = max(cat(3, Bkg, Mid, Last), [], 3);
    disp(['Tracking ' TiffName]);
    
    % preallocate arrays for whole burst
    SegmentsX = NaN(FramesTotal, nROIs, nSeg + 1); % coordinates of reconstruction, x, y, starting at head;
    SegmentsY = NaN(FramesTotal, nROIs, nSeg + 1);
    Curvature = NaN(FramesTotal, nROIs);
    Orientation = NaN(FramesTotal, nROIs);
    HeadX = NaN(FramesTotal, nROIs);
    HeadY = NaN(FramesTotal, nROIs);
    Areas = NaN(FramesTotal, nROIs);
    
    %
   
    parfor nFrame = 2:FramesTotal
        
        % preallocate arrays for each frame
        SegmentsXTmp = NaN(nROIs, nSeg+1);
        SegmentsYTmp = NaN(nROIs, nSeg+1);
        AreasTmp = NaN(nROIs, 1);
        HeadXTmp = NaN(nROIs, 1);
        HeadYTmp = NaN(nROIs, 1);
        OrientationTmp = NaN(nROIs, 1);
        CurvatureTmp = NaN(nROIs, 1);
        
        % load image, entire movie is split into 2 multipage TIFFs
        if nFrame <= Frames
            IM = LoadTiffPlane8Bit(TiffDir, TiffName, nFrame);
            
        else
            IM = LoadTiffPlane8Bit(TiffDir, TiffName2nd, nFrame-Frames);
        end
        % stabilize tracking, use cross correlation to correct the x/y
        % drift, use and erode before background subtraction to mimimize
        % well effects. 
        if (StabilizeTaps && TrialInd >= TapIndStart &&  TrialInd <= TapIndEnd && nFrame < 150 )           
            IM = AlignMultitrackerData(Bkg, IM, 150); 
            IMbkg = imerode(Bkg, ones(3,3)) - IM;
        else
            IMbkg = Bkg - IM;
        end
        
        IMfilt = imfilter(IMbkg, f); % harsh filter to find the head
        IMfiltTail = imfilter(IMbkg,  fspecial('disk', TailFiltSize)); % less harsh filter to find the tail
        ThreshStack = IMbkg > BodyThresh;
        
        for nROI = 1:nROIs
            
            % pull in the parts of the image for the ROI
            ThreshIM = ThreshStack(ROIfile(nROI,1):ROIfile(nROI,1)+ROIfile(nROI, 3), ROIfile(nROI,2):ROIfile(nROI,2)+ROIfile(nROI, 4));
            
            ROIFiltFrame = IMfilt(ROIfile(nROI,1):ROIfile(nROI,1)+ROIfile(nROI, 3), ROIfile(nROI,2):ROIfile(nROI,2)+ROIfile(nROI, 4));
            ROITailFiltFrame = IMfiltTail(ROIfile(nROI,1):ROIfile(nROI,1)+ROIfile(nROI, 3), ROIfile(nROI,2):ROIfile(nROI,2)+ROIfile(nROI, 4));
            %ROIbkgFrame = IMbkg(ROIfile(nROI,1):ROIfile(nROI,1)+ROIfile(nROI, 3), ROIfile(nROI,2):ROIfile(nROI,2)+ROIfile(nROI, 4));
            
            Stats = regionprops(ThreshIM, 'Area', 'Centroid', 'PixelIdxList');
            
            
            % find only the biggest blob
            while length(Stats) > 1;
                AreasFound = [Stats.Area];
                [Area, AreaInd] = max(AreasFound);
                if Area > maxFishSize
                    Stats(AreaInd)=[];
                else
                    Stats = Stats(AreaInd);
                    ThreshIM(:) = 0;
                    ThreshIM(Stats.PixelIdxList) = 1;
                end
            end
            
            
            if length(Stats) == 1 && Stats.Area > minFishSize
                AreasTmp(nROI) = Stats.Area;
                
                CurvatureAccumulator = 0; % keep track of summed curvature
                
                % Find the head coordinate
                [~, idx] = max(ROIFiltFrame(ThreshIM));
                [Yinds,Xinds]=find(ThreshIM);
                ptX = Xinds(idx);
                ptY = Yinds(idx);
                HeadYTmp(nROI)= ptY;
                HeadXTmp(nROI)= ptX;
                SegmentsXTmp(nROI, 1) = ptX;
                SegmentsYTmp(nROI, 1) = ptY;
                
                % Find the centroid of the blob. head -> centroid should be
                % pointing ~ head -> tail of the fish, and we will use this
                % as the 'orientation' of the fish
                
                CentX = Stats.Centroid(1);
                CentY = Stats.Centroid(2);
                
                % get the initial search angle and orientation
                sa = atan2(CentY - ptY, CentX - ptX);
                OrientationTmp(nROI) = sa;
                
                % preallocate for the angles
                Angles = NaN(nSeg, 1);
                
                for i = 1:nSeg % loop through segments
                    
                    if ~isnan(ptX) % if we didnt find a NaN previously, keep going...
                        
                        % get the arc to search for the next segment
                        
                        ang = sa-SearchAngle:pi/20:sa+SearchAngle;
                        ArcCoors = round([ptX + cos(ang)*len; ptY + sin(ang)*len])';
                        ArcCoors(ArcCoors <=0) = 1;
                        
                        % remove anything outside of the image
                        
                        xOver = ArcCoors(:,1) > size(ThreshIM, 2);
                        yOver = ArcCoors(:,2) > size(ThreshIM, 1);
                        ArcCoors(xOver, 1) = size(ThreshIM, 2);
                        ArcCoors(yOver, 2) = size(ThreshIM, 1);
                        ArcInd = sub2ind(size(ThreshIM), ArcCoors(:,2), ArcCoors(:,1));
                        
                        % find the dimmest point on the arc as the next
                        % segment coordiante
                        
                        MaxBkgVal = max(ROITailFiltFrame(ArcInd));
                        
                        if MaxBkgVal > BkgThresh
                            MaxBkgInd = ROITailFiltFrame(ArcInd) == MaxBkgVal;
                            MaxBkgInd = unique(ArcInd(MaxBkgInd));
                            [ptY, ptX] = ind2sub(size(ROIFiltFrame), MaxBkgInd);
                            
                            if length(ptY) >1
                                ptY = mean(ptY);
                                ptX = mean(ptX);
                            end
                            
                        % if we didnt find something above the BkgThresh, assume the fish isnt there, set to NaN

                        else
                            ptY = NaN;
                            ptX = NaN;
                        end
 
                        SegmentsXTmp(nROI, i+1) = ptX;
                        SegmentsYTmp(nROI, i+1) = ptY;
                        
                        % update the seach angle for the next point
                        
                        sa = atan2(SegmentsYTmp(nROI, i+1) - SegmentsYTmp(nROI, i), SegmentsXTmp(nROI, i+1) - SegmentsXTmp(nROI, i));
                        Angles(i) = sa;
                        
                        % update the curvature
                        
                        if ~isnan(ptX) && i > 1
                            CurvatureAccumulator = CurvatureAccumulator + angleDiff(Angles(i-1), Angles(i));
                        end
                        
                        
                    end
                end
                
                % Copy, and then reset the curvature accumulator
                
                CurvatureTmp(nROI) = CurvatureAccumulator;
                CurvatureAccumulator = 0;
            end
            
        end
        
        % save the values to the whole burst arrays
        
        SegmentsX(nFrame-1, :, :) = SegmentsXTmp;
        SegmentsY(nFrame-1, :, :) = SegmentsYTmp;
        Areas(nFrame-1, :) = AreasTmp';
        Orientation(nFrame-1, :) = OrientationTmp';
        HeadX(nFrame-1, :) = HeadXTmp';
        HeadY(nFrame-1, :) = HeadYTmp';
        Curvature(nFrame-1, :) = CurvatureTmp';
        
    end
    
    % write the files
    
    save(strcat(strrep(TiffName, '.tif', '_Track'), '.mat'), 'HeadX', 'HeadY', 'Orientation', 'Curvature', 'Areas', 'TiffDate');
    if SaveFishSkel
        save(strcat(strrep(TiffName, '.tif', '_Skel'), '.mat'), 'SegmentsX', 'SegmentsY');
    end
    
    % estimate the time remaining, and display
    
    EstTimeLeft = round((toc/60) * (NumTrials - nTrial));
  
    disp(['done ', num2str(nTrial), ' of ', num2str(NumTrials), ' trials... about ', num2str(EstTimeLeft), ' minutes left']);
    
    
    
end

% clean up
delete(gcp('nocreate'));
end
%clear all