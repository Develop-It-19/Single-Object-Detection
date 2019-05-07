
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#Import Dependencies
from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json		#json stands for javascript object notation and it is a standard way to pass aroung hierarchical structured data and it is not just with javascript.
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(1)wh

# We are taking our data from mirror but it has the annotations in the xml format so we take them from another link which has them in json format.

#In Python 3.x we have the pathlib library.

PATH = Path('data/pascal')

PATH.iterdir()     # This will tell that it is a generator object at a particular address.

# But to access the generator object we have to use either of the following ways:
for i in range PATH.iterdir():
     print(i)
#(or)
[i for i in range PATH.iterdir()]
#(or)
list(PATH.iterdir())

#It will return a list of Objects
#For Windows: WindowsPath
#For Linux: PosixPath

#Most External Libraries support these path objects as inputs.
#Some don't and for them you need to pass in a string.
    # a = list(PATH.iterdir())[0]
    # str(a)


trn_j = json.load((PATH/'pascal_train2007.json').open())
trn_j.key()		#Once you open up a json it becomes a Python Dictionary.

IMAGES, ANNOTATIONS, CATEGORIES = ['images', 'annotations', 'categories']
	#Handy Trick: We have put the strings inside the constants so that we can use tab complete to not make the mistakes for those constants in jupyter notebooks or in other IDE's.
trn_j[IMAGES][:5]
trn_j[ANNOTATIONS][:5]	#In this bounding box is given by [top_left(column), top_left(row), height, width]
						#This is the same way we use in PIL which is the pillow image library which has the column x row way of denoting images.
trn_j[CATEGORIES][:5]

FILE_NAME, ID, IMG_ID, CAT_ID, BBOX = 'file_name', 'id', 'image_id', 'category_id', 'bbox'

cats = dict(o[ID], o['name']) for o in trn_j[CATEGORIES]
trn_fns = dict(o[ID], o[FILE_NAME]) for o in trn_j[IMAGES]
trn_ids = [o[ID] for o in trn_j[IMAGES]]

list((PATH/'VOCdevkit'/'VOC2007').iterdir())

JPEGS = 'VOCdevkit/VOC2007/JPEGImages'

IMG_PATH = PATH/JPEGS
list(IMG_PATH.iterdir())[:5]

im0_d = trn_j[IMAGES][0]
im0_d[FILE_NAME], im0_d[ID]


trn_anno = collections.defaultdict(lambda:[])	#By using this statement we say that we need a dictionary such that even if we access the key that doesn't exist it should make the key exist by creating it and it sets itself equal to the return of the given function in this case it is lambda function.
							#We created a lambda function that takes no arguments and returns an empty list.
							#We can create our own function or can use a lambda (which is a function in place) function.

for o in trn_j[ANNOTATIONS]:
	if not o['ignore']:		#We take the ones which has 'ignore' = 0.
		bb = o[BBOX]
		bb = np.array([bb[1], bb[0], bb[1]+bb[3]-1, bb[0]+bb[2]-1])		#[top_left(row), top_left(column), bottom_right(row), bottom_right(column)]
																		#In NumPy we use row x column.
		trn_anno[o[IMG_ID]].append((bb, o[CAT_ID]))		#If the dictionary item doesn't exist yet then there is no list to append to. So we have used collections.defaultdict() to solve our problem.
len(trn_anno)

im_a = trn_anno[im0_d[ID]]
im_a

im0_a = im_a[0]
im0_a

cats[7]

trn_anno[17]

cats[15], cats[13]

def bb_hw(a):
	return np.arry(a[1], a[0], a[3]-a[1], a[2]-a[0])

im = open_image(IMG_PATH/im0_d[FILE_NAME])

def show_img(im, figsize = None, ax = None):
	if not ax: fig, ax = plt.subplots(figsize = figsize)
	ax.imshow(im)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	return ax

def draw_outline(o, lw):
	o.set_path_effects([patheffects.Stroke(linewidth = lw, foreground = 'black'), patheffects.Normal()])

def draw_rect(ax, b):
	patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill = False, edgecolor = 'white', lw = 2))
	draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz = 14):
	text = ax.text(*xy, txt, verticalalignment = 'top', color = 'white', fontsize = sz, weight = 'bold')
	draw_outline(text, 1)

ax = show_img(im)
b = bb_hw(im0_a[0])
draw_rect(ax, b)
draw_text(ax, b[:2], cats = [im0_a[1]])

def draw_im(im, ann):
	ax = show_img(im, figsize = (16, 8))
	for b, c in ann:
		b = bb_hw(b)
		draw_rect(ax, b)
		draw_text(ax, b[:2], cats[c], sz = 16)

def draw_idx(i):
	im_a = trn_anno[i]
	im = open_image(IMG_PATH/trn_fns[i])
	print(im.shape)
	draw_im(im, im_a)

draw_idx(17)



# Largest item Classifier
def get_lrg(b):
	if not b:
		raise Exception()
		b = sorted(b, key = lambda x: np.product(x[0][-2:] - x[0][:2]), reverse = True)
		return b[0]
trn_lrg_anno = {a: get_lrg(b) for a, b in trn_anno.items()}

b, c = trn_lrg_anno[23]
b = bb_hw(b)
ax = show_img(open_image(IMG_PATH/trn_fns[23]), figsize = (5, 10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz = 16)

(PATH/'tmp').mkdir(exist_ok = True)
CSV = PATH/'tmp/lrg.csv'

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'cat': [cats[trn_lrg_anno[o][1]] for o in trn_ids]}, columns = ['fn', 'cat'])
df.to_csv(CSV, index = False)

f_model = resnet34
sz = 224
bs = 64

tfms = tfms_from_model(f_model, sz, aug_tfms = transforms_side_on, crop_type = CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms = tfms)

x, y = next(iter(md.val_dl))
show_img(md.val_ds.denorm(to_np(x))[0]);

learn = ConvLearner.pretrained(f_model, md, metrics = [accuracy])
learn.opt_fn = optim.Adam

lrf = learn.lr_find(1e-5, 100)

learn.sched.plot()

learn.sched.plot(n_skip = 5, n_skip_end = 1)
	#By default it skips 10 at the start and 5 at the end.

lr = 2e-2

learn.fit(lr, 1, cycle_len = 1)

lrs = np.array([lr/1000, lr/100, lr])

learn.freeze_to(-2)

lrf = learn.lr_find(lrs/1000)
learn.sched.plot(1)

learn.fit(lrs/5, 1, cycle_len = 1)

learn.unfreeze()

learn.fit(lrs/5, 1, cycle_len = 2)

learn.save('clas_one')

learn.load('clas_one')

x, y = next(iter(md.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x, preds = to_np(x), to_np(probs)
preds = np.argmax(preds, -1)

fig, axes = plt.subplots(3, 4, figsize = (12, 8))
for i, ax in enumerate(axes.flat):
	ima = md.val_ds.denorm(x)[i]
	b = md.classes[preds[i]]
	ax = show_img(ima, ax = ax)
	draw_text(ax, (0, 0), b)
plt.tight_layout()


#pdb is the python debugger
pdb.set_trace() #To set a Breakpoint.
%debug 


# Bbox only
BB_CSV = PATH/'tmp/bb.csv'

bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
bbs = [' '.join(str(p) for p in o) for o in bb]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns = ['fn', 'bbox'])
df.to_csv(BB_CSV, index = False)

BB_CSV.open().readlines()[:5]


#### You should know these before lesson 9
		#Python Debugger
		#Matplotlib Object Oriented API

#Single Object Detection. 
f_model = resnet34
sz = 224
bs = 64

val_idxs = get_cv_idxs(len(trn_fns))

tfms = tfms_from_model(f_model, sz, crop_type = CropType.NO, tfm_y = TfmType.COORD, aug_tfms = augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms = tfms, continuous = True, val_idxs = val_idxs)
md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms = tfms_from_model(f_model, sz))

class ConcatLblDataset(Dataset):
	def __init__(self, ds, y2):
		self.ds, self.y2 = ds, y2
	def __len__(self):
		return len(self.ds)

	def__getitem__(self, i):
		x, y = self.ds[i]
		return (x, (y, self.y2[i]))

trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)

val_ds2[0][1]

md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2

x, y = next(iter(md.val_dl))
idx = 3
ima = md.val_ds.ds.denorm(to_np(x))[idx]
b = bb_hw(to_np(y[0][idx]))
b

ax = show_img(ima)
draw_rect(ax, b)
draw_text(a, b[:2], md2.classes[y[1][idx]])

head_reg4 = nn.Sequential(
	Flatten(),
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Linear(25088, 256),
	nn.ReLU(),
	nn.BatchNorm1d(256),
	nn.Dropout(0.5),
	nn.Linear(256, 4+len(cats))
	)
models = ConvnetBuilder(f_model, 0, 0, 0, custom_head = head_reg4)

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam

def detn_loss(input, target):
	bb_t, c_t = target
	bb_i, c_i = input[:, :4], input[:, 4:]
	bb_i = F.sigmoid(bb_i)*224

	return F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t)*20

def detn_l1(input, target):
	bb_t, _ = target
	bb_i = input[:, :4]
	bb_i = F.sigmoid(bb_i)*224
	return F.l1_loss(V(bb_i), V(bb_t)).data

def detn_acc(input, target):
	_, c_t = target
	c_i = input[:, 4:]
	return accuracy(c_i, c_t)

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]

learn.lr_find()
learn.sched.plot()

lr = 1e-2

learn.fit(lr, 1, cycle_len = 3, use_clr = (32, 5))


#----------------------------------------------------------





