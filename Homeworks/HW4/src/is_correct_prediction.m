function [correct] = is_correct_prediction(actual,prediction)

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
