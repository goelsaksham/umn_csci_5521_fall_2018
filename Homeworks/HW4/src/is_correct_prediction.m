function [correct] = is_correct_prediction(actual,prediction)
% this function is to wether the actual label is the same as what is
% predicted from the network.

correct = 0;
if (actual(1) < 0 && prediction(1) < 0)
    if actual(2) > 0 && prediction(2) > 0
        correct = 1;
    end
end
if actual(1) > 0 && prediction(1) > 0
    if actual(2) < 0 && prediction(2) < 0
        correct = 1;
    end
end
end

