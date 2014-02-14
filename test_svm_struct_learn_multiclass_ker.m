function test_svm_struct_learn_multiclass_ker
% TEST_SVM_STRUCT_LEARN
%   Test function for SVM_STRUCT_LEARN(). It shows how to use
%   SVM-struct to learn a standard linear SVM while using the generic
%   kernel interface.

  randn('state',0) ;
  rand('state',0) ;

  % ------------------------------------------------------------------
  %                                                      Generate data
  % ------------------------------------------------------------------
  
  samples_per_class = 50 ;
  labels = num2cell(kron([1 2 3]',ones(samples_per_class,1))) ;
  
  patterns = cell(samples_per_class*3,1) ;
  for i = 1:samples_per_class*3
    switch labels{i}
      case 1, patterns{i} = [1 1] + rand(1,2) ;
      case 2, patterns{i} = [-2 -2] + rand(1,2) ;
      otherwise
        if rand>0.5
          patterns{i} = [-1 1] + 0.3 * randn(1,2) ;
        else
          patterns{i} = [1 -1] + 0.3 * randn(1,2) ;
        end
    end
  end
  
  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------
  
  fprintf('SVM struct: Multiclass example with non-linear kernel\n') ;
  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.lossFn = @lossCB;
  parm.constraintFn  =@constraintCB ;
  parm.kernelFn = @kernelCB ;
  parm.verbose = 0 ;
  tic_learning = tic ;
  model = svm_struct_learn(' -c 5.0 -o 2 -v 1 -t 4 -w 3', parm) ;
  tic_learning = toc(tic_learning) ;
  fprintf('Approx time of learning with %d samples: %8.3fs\n', ...
      samples_per_class*3, tic_learning) ;

  % ------------------------------------------------------------------
  %                                                   Run SVM struct 2
  % ------------------------------------------------------------------
  
  fprintf(['\n\n\nSVM struct: Multiclass example with non-linear kernel\n' ...
    '!!! Using and incomplete definition of the kernel function !!!\n']) ;
  parm.kernelFn = @incomplete_kernelCB ;
  incomplete_model = svm_struct_learn(' -c 5.0 -o 2 -v 1 -t 4 -w 3', parm) ;
  fprintf('\n\n\n') ;
  
  % ------------------------------------------------------------------
  %                                                              Plots
  % ------------------------------------------------------------------
  
  models = [model;incomplete_model] ;
  parm = repmat(parm,2,1) ; parm(1).kernelFn = @kernelCB ;
  for i = 1:2
    figure(i) ; clf ; hold on ;
    x = cat(1,patterns{:}) ;
    y = cat(1,labels{:}) ;
    plot(x(y==1, 1), x(y==1, 2), 'go') ;
    plot(x(y==2, 1), x(y==2, 2), 'ro') ;
    plot(x(y==3, 1), x(y==3, 2), 'bo') ;
    x_range = [min(x(:, 1))-0.1 max(x(:, 1))+0.1] ; xlim(x_range) ; 
    y_range = [min(x(:, 2))-0.1 max(x(:, 2))+0.1] ; ylim(y_range) ;

    % Visualize decision boundaries
    x = x_range(1) : 0.1 : x_range(2) ;
    y = y_range(1) : 0.1 : y_range(2) ;
    [x, y] = meshgrid(x, y) ;
    x = [x(:) y(:)] ;
    y = zeros(size(x, 1), 1) ;
    tic_inference = tic;
    for j = 1:size(x, 1)
      y(j) = inferenceCB(parm(i), models(i), x(j,:)) ;
    end
    tic_inference = toc(tic_inference);
    plot(x(y==1, 1), x(y==1, 2), 'g.') ;
    plot(x(y==2, 1), x(y==2, 2), 'r.') ;
    plot(x(y==3, 1), x(y==3, 2), 'b.') ;
    
    fprintf('Approx time of inference per sample: %8.3fs\n', tic_inference / ...
      size(x,1));
  end
  
end

% --------------------------------------------------------------------
%                                                SVM struct callbacks
% --------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
% loss function delta(y, ybar)
  delta = double(y ~= ybar) ;
  if param.verbose
    fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
  end
end

function k = incomplete_kernelCB(param, x, y, xp, yp)
  sigma = 0.25 ;
  switch y
    case 3, k = exp(- norm(x-xp) / sigma / 2) ;
    otherwise, k = xp * x' / 4 ;
  end
end

function k = kernelCB(param, x, y, xp, yp)
  if y == yp
    k = incomplete_kernelCB(param, x, y, xp, yp) ;
  else
    k = 0 ;
  end
end

function yhat = constraintCB(param, model, x, y)
% Compute the augmented loss inference implicitly.
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  Y = [1 2 3] ; cardinal_Y = numel(Y) ;
  n = size(model.svPatterns, 2) ;
  verbose = param.verbose ;  param.verbose = 0 ;
  if n == 0
    Y_minus_y = setdiff(Y,y) ;
    yhat = Y_minus_y(randi(cardinal_Y - 1)) ;
  else
    augmented_loss = zeros(1,cardinal_Y) ;
    for i = Y
      k = cellfun(param.kernelFn, cell(1,n), model.svPatterns, ...
                  model.svLabels, repmat({x},1,n), repmat({i},1,n)) ;
      augmented_loss(i) = lossCB(param,y,i) + k * model.alpha ;
    end
    [maxim idx] = max(augmented_loss) ;
    yhat = Y(idx) ;
  end
  param.verbose = verbose ;
  
  if param.verbose
    fprintf(['yhat = violmargin([1: %8.3f, 2: %8.3f, 3: %8.3f], ' ...
      '[%8.3f,%8.3f], %3d) = %3d\n'], augmented_loss, x, y, yhat) ;
  end
end

function yhat = inferenceCB(param, model, x)
% Predict yhat from an instance x.
% yhat: argmax_y <psi(x,y), w>
  Y = [1 2 3] ; cardinal_Y = numel(Y);
  n = size(model.svPatterns, 2) ;
  verbose = param.verbose ;  param.verbose = 0 ;
  discriminative_funct = zeros(1,cardinal_Y) ;
  for i = Y
    k = cellfun(param.kernelFn, cell(1,n), model.svPatterns, model.svLabels, ...
                repmat({x},1,n), repmat({i},1,n)) ;
    discriminative_funct(i) = k * model.alpha ;
  end
  [maxim idx] = max(discriminative_funct) ;
  yhat = Y(idx) ;
  param.verbose = verbose ;
  
  if param.verbose
    fprintf(['yhat = inference([1: %8.3f, 2: %8.3f, 3: %8.3f], ' ...
      '[%8.3f,%8.3f] = %3d\n'], discriminative_funct, x, yhat) ;
  end
end
