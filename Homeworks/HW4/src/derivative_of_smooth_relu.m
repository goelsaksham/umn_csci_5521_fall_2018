function [y] = derivative_of_smooth_relu(yhat)
% This function is used to find the derivative of the smooth relu function.
% However the derivative is actually with respect to the yhat value (the
% output of the smooth_relu function). The actual derivative is different
% and to avoid recalculating the value by smooth_relu function again, we
% try to use the already computed value to calculate the derivative during
% backpropogation.
y = ((exp(10*yhat) - 1) / exp(10*yhat));
end

