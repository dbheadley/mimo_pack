# Timing preprocessing related functions
# Author: Drew Headley
# Created: 2024-07-08

function [tOthNew, qualMetrics] = datdattimealign(fPathOth,fPathRef,chanOth,chanRef)
%% datdattimealign
% Synchronizes the timing of two dat files

%% Syntax
%# timesOthNew = datdattimealign(fPathOth,fPathRef,chanOth,chanRef)

%% Description
% A synchonization pulse train shared between two dat files, fPathOth and
% fPathRef, is used to synchronize their times. The pulse train signals are
% on chanOth and chanRef, respectively. The _t file of fPathOth is changed
% to reflect the timestamps in fPathRef. The original _t file for fPathOth
% is overwritten with the new timestamps. Thus, the two files can now be
% called with the same time reference.

%% INPUT
%  * fPathOth - a string, the name of the binary file whose _t file will
% be synchronized.
%  * fPathRef - a string, the name of the binary file that will be
% used as the timing reference.
%  * chanOth - an interger or string, the channel carrying the
%  synchronization signal in fPathOth.
%  * chanRef - an interger or string, the channel carrying the
%  synchronization signal in fPathRef.


%% OUTPUT
% * tOthNew - an array of number, the new timestamps for fPathOth that are
% aligned with those in fPathRef
% * qualMetrics - a structure, metrics for the quality of the time
% alignment.
%       -MedianDT: the median time step based on interpolation
%       -MinDT: the shortest time step based on interpolation
%       -MaxDT: the longest time step based on interpolation
%       -InterpPercent: the percentage of time points aligned via interpolation

%% Example

%% Executable code

% load synchrony channels
syncOth = readdat(fPathOth,'selchans', chanOth);
syncRef = readdat(fPathRef,'selchans', chanRef);

othInfo = datinfo(fPathOth);

% reference time points
refTPts = syncRef.tPts{1};

% convert to boolean
syncOth = zscore(syncOth.traces{1})>0;
syncRef = zscore(syncRef.traces{1})>0;

% get pulse duration sequence
seqOth = regionprops(syncOth,{'Area' 'PixelIdxList'});
seqRef = regionprops(syncRef,{'Area' 'PixelIdxList'});

% align pulse sequences
pulseAligns = MeasureAlignment(vertcat(seqRef.Area),vertcat(seqOth.Area),30);
pulseEdgesOth = cellfun(@(x)x(1),{seqOth(pulseAligns(:,2)).PixelIdxList});
pulseEdgesRef = cellfun(@(x)x(1),{seqRef(pulseAligns(:,1)).PixelIdxList});
pulseEdgesOth = pulseEdgesOth(:);
pulseEdgesRef = pulseEdgesRef(:);

% if pulse is at the beginning, remove it because its start is ambiguous
if (pulseEdgesRef(1)==1)
    pulseEdgesRef(1) = [];
    pulseEdgesOth(1) = [];
end

% match pulse edges to time points
tPulseEdgesRef = refTPts(pulseEdgesRef);

% generate new time points for oth that are aligned with ref
tOthNew = interp1(pulseEdgesOth', tPulseEdgesRef', ...
                  pulseEdgesOth(1):pulseEdgesOth(end));

% calculate quality metrics based on interpolation results
dTList = diff(tOthNew);
medDT = median(dTList);
minDT = min(dTList);
maxDT = max(dTList);
qualMetrics.MedianDT = medDT;
qualMetrics.MinDT = minDT;
qualMetrics.MaxDT = maxDT;
qualMetrics.InterpPercent = length(tOthNew)/length(syncOth);

% add times for the beginning and end of the file
tOthNew = [(-((pulseEdgesOth(1)-1):-1:1)*medDT)+tOthNew(1) tOthNew];

remTPts = length(syncOth)-length(tOthNew);
tOthNew = [tOthNew tOthNew(end) + ((1:remTPts)*medDT)];


% overwrite old _t file for oth with new time points
othFID = fopen(othInfo.TFile,'w');
fwrite(othFID,tOthNew,'double');
fclose(othFID);



function seqPairs = MeasureAlignment(ser1, ser2, matchLen)
    % seqPairs are the matched entries in ser1 and ser2. For every matched
    % pair the index of the entry in ser1 is given in column 1, and the
    % corresponding entry in ser2 is given in column 2.
    % matchLen should be an even number
    
    if rem(matchLen,2) == 1
        error('Match Length must be an even number');
    end
    
    serLen1 = length(ser1)+matchLen;
    serLen2 = length(ser2)+matchLen;
    
    % pad with infs to handle edge effects
    ser1 = [inf(matchLen,1); zscore(ser1(:)); inf(matchLen,1)];
    ser2 = [inf(matchLen,1); zscore(ser2(:)); inf(matchLen,1)];

    serPt1 = matchLen+1;
    serPt2 = matchLen+1;
    
    seqPairs = [];
    waitH = waitbar(0,'Aligning pulse sequences');
    % matching loop
    while (serPt1 <= serLen1)
        waitbar(serPt1/serLen1,waitH);
        
        serOff2 = 0;
        while ((serPt2+serOff2) <= serLen2)
            
            seqDiff = bwareaopen(abs(ser1(serPt1+[-matchLen:matchLen])...
                                     -ser2((serPt2+serOff2)+[-matchLen:matchLen]))<1,matchLen);
            if seqDiff(matchLen+1)
                serPt2 = serPt2 + serOff2;
                seqPairs(end+1,:) = [serPt1 serPt2];
                serPt2 = serPt2 + 1;
                break;
            else
                serOff2 = serOff2 + 1;
            end
        end
        serPt1 = serPt1 + 1;
    end
    
    close(waitH);      
    seqPairs = seqPairs-matchLen;
end            
            
