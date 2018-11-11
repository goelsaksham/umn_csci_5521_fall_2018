function [assignments, centers, StepCount] = newKmeans(data, InitialCenters)
%DOKMEANS Summary of this function goes here
%   Detailed explanation goes here

CurrentCenters = InitialCenters;
number_of_centers = size(InitialCenters);
OldCenters = zeros(number_of_centers);
% number_of_features = size(data, 2);

clusters = zeros(size(data, 1), size(data, 2)+1);
StepCount = 0;
while sum(sum(OldCenters ~= CurrentCenters)) > 0
    for i=1:size(data, 1)
        minimum_distant_center = 1;
        minimum_distance = findDistance(CurrentCenters(minimum_distant_center, :), data(i, :));
        for j=1:number_of_centers(1, 1)
            center = CurrentCenters(j, :);
            distance = findDistance(center, data(i, :));
            if distance < minimum_distance
                minimum_distant_center = j;
                minimum_distance = distance;
            end
        end
        clusters(i, :) = [data(i, :) minimum_distant_center];
    end
    
    OldCenters = CurrentCenters;
    for k=1:number_of_centers(1,1)
        cluster_of_center = clusters(logical(clusters(:, end) == k), :);
        cluster_of_center = cluster_of_center(:, 1:end-1);
        cluster_average = mean(cluster_of_center);
        CurrentCenters(k, :) = cluster_average;
    end
    StepCount = StepCount + 1;
end

centers = CurrentCenters;
assignments = clusters;

