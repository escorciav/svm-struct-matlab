function test_svm_struct_multiclass_learn
% TEST_SVM_STRUCT_MULTICLASS_LEARN
%   Test function for SVM_STRUCT_LEARN(). It shows how to use SVM-struct to
%   a multi-class linear SVM.

  randn('state',0) ;
  rand('state',0) ;

  nsamples = 120 ;
  parm.nclass = 3 ;
  parm.featsize = 2 ;
  parm.dimension = (parm.featsize+1) * parm.nclass ;
  parm.C = 0.01 ;
  parm.verbose = 0 ;
  parm.color= 'grb' ;
  
  % ------------------------------------------------------------------
  %                                                      Generate data
  % ------------------------------------------------------------------

  patterns = cell(1,nsamples) ;  labels = patterns ;
  idx_class = floor(linspace(1,nsamples,parm.nclass+1)) ;
  idx_class = 0.5*(idx_class(2:end)+idx_class(1:end-1)) ;
  for i=1:nsamples
    [dummy labels{i}] = min(pdist2(idx_class',i)) ;
    switch labels{i}
      case 1
        patterns{i} = [[2;2]+[0.5;0.5].*randn(parm.featsize,1);1] ;
      case 2
        patterns{i} = [[-2;-2]+[1.5;1.5].*randn(parm.featsize,1);1] ;
      otherwise
        patterns{i} = [[-2;2]+[1.5;1.5].*randn(parm.featsize,1);1] ;
    end
  end

  parm.featsize = parm.featsize + 1 ;  %include bias
  
  % ------------------------------------------------------------------
  %                                                          Plot Data
  % ------------------------------------------------------------------

  figure(1) ; clf ; subplot(1,2,1) ; hold on ; subplot(1,2,2) ; hold on ;
  x = [patterns{:}] ;
  y = [labels{:}] ;
  for i=1:parm.nclass
    subplot(1,2,1), plot(x(1, y==i), x(2, y==i), [parm.color(i) 'o'], ...
                         'Linewidth', 1) ;
    subplot(1,2,2), plot(x(1, y==i), x(2, y==i), [parm.color(i) 'o'], ...
                         'Linewidth', 1) ;
  end
  x_lim = [floor(min(x(1,:)))-0.5 ceil(max(x(1,:)))+0.5] ;  
  subplot(1,2,1); xlim(x_lim); subplot(1,2,2); xlim(x_lim) ;
  y_lim = [floor(min(x(2,:)))-0.5 ceil(max(x(2,:)))+0.5] ;  
  subplot(1,2,1); ylim(y_lim); subplot(1,2,2); ylim(y_lim) ;

  % ------------------------------------------------------------------
  %                                                     Run SVM struct
  % ------------------------------------------------------------------

  parm.patterns = patterns ;
  parm.labels = labels ;
  parm.lossFn = @lossCB ;
  parm.constraintFn  = @slack_constraintCB ;
  parm.featureFn = @featureCB ;
  % Learning
  ssvm_multiclass = svm_struct_learn([' -c ',num2str(parm.C),' -o 1 -v 1 '], ...
                                     parm) ;
  w = reshape(ssvm_multiclass.w,parm.featsize,[]) ;

  [yhat score] = classifyCB(parm,ssvm_multiclass,cat(2,parm.patterns{:})) ;

  % ------------------------------------------------------------------
  %                                   Plot Result Multiclass Struc SVM
  % ------------------------------------------------------------------
  % w is the normal vector of SVM hyperplane [w1 w2 ... wn b] then w*[x 1]=0
  % then in 2D w*[x y 1]=0 -> y = -b/w2 - w1/w2 x
  % margin: 1 / ||(x,y)||  
  % bias: |b| / ||(x,y)||

  for i=1:parm.nclass
    y_plane = -1 * w(3,i) / w(2,i) - x_lim * w(1,i) / w(2,i) ;
    subplot(1,2,1), set(line(x_lim, y_plane), 'color', parm.color(i)) ;
  end
  title('Multiclass Structure SVM', 'fontSize', 14)
  
  % ------------------------------------------------------------------
  %                                              Run Binary SVM struct
  % ------------------------------------------------------------------

  ssvm_binary = cell(parm.nclass, 1) ;  w_bin = ssvm_binary ;
  parm_binary.nclass = parm.nclass ;
  parm_binary.dimension = parm.featsize ;
  parm_binary.C = 0.01 ;
  parm_binary.verbose = 0 ;
  for i = 1:parm_binary.nclass
    parm_binary.patterns = parm.patterns ;
    parm_binary.labels = cellfun(@(x) double(x==i) * 2 - 1,labels, ...
                                 'uniformoutput',0);
    parm_binary.lossFn = @lossCB ;
    parm_binary.constraintFn  = @binary_constraintCB ;
    parm_binary.featureFn = @binary_featureCB ;
    % Learning
    ssvm_binary{i} = svm_struct_learn([' -c ', num2str(parm_binary.C), ...
                                       ' -o 1 -v 0 '], parm_binary) ;
    w_bin{i} = ssvm_binary{i}.w ;
  end
  w_bin=reshape(cat(1,w_bin{:}),parm_binary.nclass,[]) ;

  score_binary = zeros(nsamples, parm_binary.nclass) ;
  yhat_perbinary = score_binary ;
  for i = 1:parm_binary.nclass
    [yhat_perbinary(:,i) score_binary(:,i)] = ...
                         binary_classifyCB(parm_binary, ssvm_binary{i}, ...
                                           cat(2, parm_binary.patterns{:})) ;
  end
  [dummy yhat_binary] = max(score_binary, [], 2) ;

  % ------------------------------------------------------------------
  %                                   Plot Result OVA Binary Struc SVM
  % ------------------------------------------------------------------

  for i = 1:parm.nclass
    y_plane = -1 * w_bin(3,i) / w_bin(2,i) - x_lim * w_bin(1,i) / w_bin(2,i);
    subplot(1,2,2), set(line(x_lim, y_plane), 'color', parm.color(i), ...
                        'LineStyle','--') ;
  end
  title('One VS All Binary SVM', 'fontSize', 14)
  
  % ------------------------------------------------------------------
  %                                              Classification Errors
  % ------------------------------------------------------------------
  
  ssvm_e = yhat ~= cat(1, parm.labels{:}) ;
  ssvm_e_binary = yhat_binary ~= cat(1, parm.labels{:}) ;
  subplot(1,2,1), legend({['multiclass SSVM errors: ' ...
                           num2str(sum(ssvm_e)) '/120']}, 'fontsize', 11) ;
  subplot(1,2,2), legend({['OVA binary SVM errors: ' ...
                           num2str(sum(ssvm_e_binary)) '/120']}, ...
                           'fontsize', 11) ;
  x = [patterns{:}] ;  y = [labels{:}]';
  for i=1:parm.nclass
    subplot(1,2,1),plot(x(1, y==i & ssvm_e), x(2,y==i & ssvm_e), ...
                        [parm.color(i) '*'], 'Linewidth', 1) ;
    subplot(1,2,2),plot(x(1, y==i & ssvm_e_binary), ...
                        x(2,y==i & ssvm_e_binary), [parm.color(i) '*'], ...
                        'Linewidth', 1) ;
  end

end

% -----------------------------------------------------------------
%                                    Multiclass SVM struct callbacks
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
% Loss function
  delta = double(y ~= ybar) ;
  if param.verbose
    fprintf('delta = loss(%3d, %3d) = %f\n', y, ybar, delta) ;
  end
end

function psi = featureCB(param, x, y)
% Joint feature map Psi(x,y)
  psi = zeros(param.dimension, size(x,2)) ;
  psi((y-1)*param.featsize+1:y*param.featsize,:) = x;  psi = sparse(psi);
  if param.verbose
    fprintf(['w = psi([%8.3f,%8.3f], %3d) = ' ...
             '[%8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f]\n'], x, y, ...
             full(psi(1)), full(psi(2)), full(psi(3)), full(psi(4)), ...
             full(psi(5)), full(psi(6))) ;
  end
end

function yhat = slack_constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
  psi = full( featureCB(param,x,y) ) ;
  score_y = dot(psi, model.w) ;

  bestclass = -1 ; first = 1 ; bestscore = -inf ;
  for class = 1:param.nclass
    psi = full( featureCB(param, x, class) ) ;
    score_yhat = dot(psi,model.w) ;
    score = (y ~= class) * (1 - score_y + score_yhat);
    if score > bestscore || first
      bestscore = score;  bestclass = class;  first = 0;
    end
  end
  yhat = bestclass;
end

function [yhat score] = classifyCB(param, model, x)
% Predict the class of an array of instances x according to the SSVM model
  score = zeros(size(x,2), param.nclass) ;
  for class = 1:param.nclass
    psi = full( featureCB(param,x,class) ) ;
    score(:,class) = sum( bsxfun(@times, model.w, psi) )' ;
  end
  [dummy yhat] = max(score, [], 2);
end

% -----------------------------------------------------------------
%                                        Binary SVM struct callbacks
% ------------------------------------------------------------------

function psi = binary_featureCB(param, x, y)
  psi = sparse(y*x/2) ;
  if param.verbose
    fprintf('w = psi([%8.3f,%8.3f], %3d) = [%8.3f, %8.3f]\n', ...
            x, y, full(psi(1)), full(psi(2))) ;
  end
end

function yhat = binary_constraintCB(param, model, x, y)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end
  if param.verbose
    fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
            model.w, x, y, yhat) ;
  end
end

function [yhat score] = binary_classifyCB(param, model, x)
% Predict the class of an array of instances x according to the SSVM model
  score = sum( bsxfun(@times, model.w, x) ) ;
  yhat = (score > 0) * 2 - 1;
end