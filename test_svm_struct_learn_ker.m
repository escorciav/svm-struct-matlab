function test_svm_struct_learn_ker
% TEST_SVM_STRUCT_LEARN
%   Test function for SVM_STRUCT_LEARN(). It shows how to use
%   SVM-struct to learn a standard linear SVM while using the generic
%   kernel interface.

  randn('state',0) ;
  rand('state',0) ;

  % ------------------------------------------------------------------
  %                                                      Generate data
  % ------------------------------------------------------------------

  th = pi/3 ;
  c = cos(th) ;
  s = sin(th) ;

  patterns = {} ;
  labels = {} ;
  for i=1:100
    patterns{i} = diag([2 .5]) * randn(2, 1) ;
    labels{i}   = 2*(randn > 0) - 1 ;
    patterns{i}(2) = patterns{i}(2) + labels{i} ;
    patterns{i} = [c -s ; s c] * patterns{i}  ;
  end

  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.lossFn = @lossCB;
  parm.constraintFn  =@constraintCB ;
  parm.kernelFn = @kernelCB ;
  parm.verbose = 1 ;
  model = svm_struct_learn(' -c 1.0 -o 1 -v 1 -t 4 -w 3', parm) ;
  w = cat(2, model.svPatterns{:}) * ...
      (model.alpha .* cat(1, model.svLabels{:})) / 2 ;

  % ------------------------------------------------------------------
  %                                                              Plots
  % ------------------------------------------------------------------

  figure(1) ; clf ; hold on ;
  x = [patterns{:}] ;
  y = [labels{:}] ;
  plot(x(1, y>0), x(2,y>0), 'g.') ;
  plot(x(1, y<0), x(2,y<0), 'r.') ;
  set(line([0 w(1)], [0 w(2)]), 'color', 'y', 'linewidth', 4) ;
  xlim([-3 3]) ;
  ylim([-3 3]) ;
  set(line(10*[w(2) -w(2)], 10*[-w(1) w(1)]), ...
      'color', 'y', 'linewidth', 2, 'linestyle', '-') ;
  axis equal ;
  set(gca, 'color', 'b') ;
  w
  
  % ------------------------------------------------------------------
  %                                                    Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.lossFn = @lossCB;
  parm.constraintFn  =@constraintCB_implicit ;
  parm.kernelFn = @kernelCB ;
  parm.verbose = 0 ;
  model = svm_struct_learn(' -c 1.0 -o 2 -v 1 -t 4 -w 3', parm) ;
  w = cat(2, model.svPatterns{:}) * ...
      (model.alpha .* cat(1, model.svLabels{:})) / 2
  
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

function k = kernelCB(param, x,y, xp, yp)
  k = x' * xp * y * yp / 4 ;
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>

  % the kernel is linear, get a weight vector back
  if size(model.svPatterns, 2) == 0
    w = zeros(size(x)) ;
  else
    w = [model.svPatterns{:}] * (model.alpha .* [model.svLabels{:}]') / 2 ;
  end
  if dot(y*x, w) > 1, yhat = y ; else yhat = -y ; end
  if param.verbose
    fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
            w, x, y, yhat) ;
  end
end

function yhat = constraintCB_implicit(param, model, x, y)
% Compute the augmented loss inference implicitly.
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  Y = [-1 1];
  n = size(model.svPatterns, 2);
  verbose = param.verbose;  param.verbose = 0;
  if n == 0
    yhat = -y;
  else
    augmented_loss = zeros(1,2);
    for i = Y
      k = cellfun(@kernelCB, cell(1,n), model.svPatterns, model.svLabels, ...
                  repmat({x},1,n),repmat({i},1,n));
      augmented_loss(0.5*i+1.5) = lossCB(param,y,i) + k * model.alpha;
    end
    [maxim idx] = max(augmented_loss);
    yhat = Y(idx);
  end
  param.verbose = verbose;
  if param.verbose
    fprintf(['yhat = violmargin([-1: %8.3f, 1: %8.3f], [%8.3f,%8.3f], %3d)' ...
      '= %3d\n'], augmented_loss, x, y, yhat) ;
  end
end
