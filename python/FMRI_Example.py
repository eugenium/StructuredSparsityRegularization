"""
Decoding scissor and scramblepixel with proximal K-support and isotropic TV
"""
# author: Eugene Belilovsky
#

from nilearn.input_data import NiftiMasker
from sklearn.linear_model.base import center_data
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
from SolveTVK.Ksupport import Ksupport
from SolveTVK.ksuptv_solver import tvksp_solver
KSup=Ksupport()

def _crop_mask(mask):
    """Crops input mask to produce tighter (i.e smaller) bounding box with
    the same support (active voxels)"""
    idx = np.where(mask)
    i_min = max(idx[0].min() - 1, 0)
    i_max = idx[0].max()
    j_min = max(idx[1].min() - 1, 0)
    j_max = idx[1].max()
    k_min = max(idx[2].min() - 1, 0)
    k_max = idx[2].max()
    return mask[i_min:i_max + 1, j_min:j_max + 1, k_min:k_max + 1]
    
### Load haxby dataset ########################################################
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby('/home/eugene/Documents/')

### Load Target labels ########################################################
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")


### split data into train and test samples ####################################
n_train=6
target = labels['labels']
condition_mask = np.logical_or(target == "scissors", target == "scrambledpix")
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= n_train)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > n_train)

# make X (design matrix) and y (response variable)
import nibabel
niimgs  = nibabel.load(data_files.func[0])
X_train = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask_train],
                        niimgs.get_affine())
y_train = target[condition_mask_train]
X_test = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask_test],
                        niimgs.get_affine())
y_test = target[condition_mask_test]

y_train[y_train=='scissors']=1
y_train[y_train=='scrambledpix']=-1
y_train=np.array(y_train.astype('double'))

y_test[y_test=='scissors']=1
y_test[y_test=='scrambledpix']=-1
y_test=np.array(y_test.astype('double'))



masker = NiftiMasker(mask_strategy='epi',standardize=True)
                        
X_train = masker.fit_transform(X_train)
X_test  = masker.transform(X_test)

mask = masker.mask_img_.get_data().astype(np.bool)
mask= _crop_mask(mask)
background_img = mean_img(data_files.func[0])

X_train, y_train, _, y_train_mean, _ = center_data(X_train, y_train, fit_intercept=True, normalize=False,copy=False)
X_test-=X_train.mean(axis=0)
X_test/=np.std(X_train,axis=0)
alpha=1
ratio=0.5
k=200


solver_params = dict(tol=1e-6, max_iter=5000,prox_max_iter=100)

init=None
w,obj,init=tvksp_solver(X_train,y_train,alpha,ratio,k,mask=mask,init=init,loss="logistic",verbose=1,**solver_params)
coef=w[:-1]
intercept=w[-1]    
coef_img=masker.inverse_transform(coef)
y_pred=np.sign(X_test.dot(coef)+intercept)
accuracie=(y_pred==y_test).mean()


print 
print "Results"
print "=" * 80

plot_stat_map(coef_img, background_img,
              title="accuracy %g%% , k: %d, alpha: %d, ratio: %.2f" % (accuracie,k,alpha,ratio),
              cut_coords=(37, -48, -15),threshold=1e-7)


print "Number of train samples : %i" % condition_mask_train.sum()
print "Number of test samples  : %i" % condition_mask_test.sum()
print "Classification accuracy : %g%%" % accuracie
print "_" * 80
r=KSup.findR(coef,k)[0]
coef_kr=coef.copy()
coef_k=coef.copy()
coef_kr[np.abs(coef)<np.sort(np.abs(coef))[::-1][k-r]]=0
coef_k[np.abs(coef)<np.sort(np.abs(coef))[::-1][k]]=0
coef_kr_img=masker.inverse_transform(coef_kr)
coef_k_img=masker.inverse_transform(coef_k)
plot_stat_map(coef_kr_img, background_img,
              title="acc %g%% , k: %d, alpha: %d, ratio: %.2f|%d of %d voxels" % (accuracie,k,alpha,ratio,k-r,len(coef_kr)),
              cut_coords=(37, -48, -15),threshold=1e-30)



