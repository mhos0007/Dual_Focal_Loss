classdef dfl_loss_function < nnet.layer.ClassificationLayer

    methods
        function this = dfl_loss_function(name)
            % Set layer name.
            this.Name = name;

            % Set layer description.
            this.Description = 'Dual Focal Loss';
        end
        
        function loss = forwardLoss(this, Y, T)
                
                %We used Mean Absolute Error (MAE) as the "forward loss" in
                %our paper to evaluate the loss after each iteration. This 
                %is to compare DFL with other loss functions on the same benchmark.
                
                loss_mae = abs(T-Y);
                loss = mean(loss_mae(:)); %the mean MAE loss
                
                %In case you would like to use DFL as the "forward loss",
                %the expression is as follows (uncomment the lines when using):
                
%                 thres = 0.001;
%                 alpha = 1;
%                 beta = 1;
%                 gamma = 1;
%                 rho = 1;

%                 Y_on_positive_classes = this.prevent_approaching_zero(Y, thres);
%                 Y_on_negative_classes = this.prevent_approaching_zero(rho-Y, thres);

%                 loss_dfl = - (T.*log(Y_on_positive_classes) + (1-T) .* log(Y_on_negative_classes) - alpha.*abs(T-Y).^gamma);

%                 loss = mean(loss_dfl(:)); %the mean DFL loss
                
        end
        
        function dX = backwardLoss(this, Y, T)
            
                numObservations = size(Y, 4) * size(Y, 1) * size(Y, 2);
                thres = 0.001;
                alpha = 1;
                beta = 1;
                gamma = 1;
                rho = 1;
                Y_on_positive_classes = this.prevent_approaching_zero(Y, thres);
                Y_on_negative_classes = this.prevent_approaching_zero(rho-Y, thres);
                                
                %In the following equation, the term "(2.*T-1)" is equivalent 
                %to the derivative of abs(T-Y). The reason of using "1.*(2.*T-1)"
                %is to prevent the denominator approaching zero, and thus
                %providing a numerical stability.

                dX = (-T./(Y_on_positive_classes) + beta.*(1-T)./(Y_on_negative_classes) ...
                    - alpha.*(2.*T-1).^gamma) .* (1./numObservations);
                
                %In case you are wondering, the derivative of DFL, keeping
                %the derivative of abs(T-Y), would look like the following.
                %If you use this version, you might notice numerical
                %unstability in the training due to "exploding gradient",
                %caused by the derivative of abs(T-Y). Although the
                %function: prevent_approaching_zero() can be used, with a very small
                %threshold value, to avoid this, we prefer simply using
                %the equivalent term: "(2.*T-1)".
                
%                 dX = (-T./(Y_on_positive_classes) + beta.*(1-T)./(Y_on_negative_classes) ...
%                     - alpha.*(abs(T-Y)./(T-Y)).^gamma) .* (1./numObservations);
        end

        function out = prevent_approaching_zero(this, z, thres)

                %This function prevents the denominator approach zero; otherwise 
                %the loss will be huge and delay the convergence. We used 0.001 
                %as a threshold value. However, a different threshold value may 
                %work better for a different problem domain.

                z(z < thres) = thres;
                out = z;
        end
                
    end
end