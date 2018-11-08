function [ConfusionMatrix] = GetConfusionMatrix(TrueLabels, Assignments)
%   GETCONFUSIONMATRIX is my implementation of computing the confusion
%   matrix for the k-means initialization. This confusion matrix represents
%   the cluster centers and the number of values for each class which
%   are assigned to that particular cluster. Consider if the number of
%   unique clusters is M and the number of unique classes is N, then the output
%   matrix would be of size M X N.

% Parameters:
%   TrueLabels - The true class labels for all the samples
%   AssignedLabels - The assigned cluster labels for all the samples

classLabels = sort(unique(TrueLabels));
clusterLabels = sort(unique(Assignments));
ConfusionMatrix = zeros(size(clusterLabels, 1), size(classLabels, 1));

for cluster_label_index = 1:size(clusterLabels, 1)
    for class_label_index = 1:size(classLabels, 1)
        ConfusionMatrix(cluster_label_index, class_label_index) = sum(TrueLabels(Assignments == clusterLabels(cluster_label_index)) == classLabels(class_label_index));
    end
end

end
