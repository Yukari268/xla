from multiprocessing.dummy import Array
from modules.dependencies.model import create_model
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
from modules.dependencies.align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)    
class DataLoader():
    def __init__(self) -> None:
        pass
    def load_metadata(self,path):
        metadata = []
        for i in sorted(os.listdir(path)):
            for f in sorted(os.listdir(os.path.join(path, i))):
                # Check file extension. Allow only jpg/jpeg' files.
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
                    metadata.append(IdentityMetadata(path, i, f))
        return np.array(metadata)

    def load_image(self,path):
        img = cv2.imread(path, 1)
        # OpenCV loads images with color channels
        # in BGR order. So we need to reverse them
        return img[...,::-1]
class Loader():
    def __init__(self, dataloader : DataLoader) :
        self.nn4_small2_pretrained = create_model()
        self.nn4_small2_pretrained.load_weights('modules/dependencies/weights/nn4.small2.v1.h5')


        self.metadata = dataloader.load_metadata('./pictures')

        # Initialize the OpenFace face alignment utility
        self.alignment = AlignDlib('modules/dependencies/models/shape_predictor_68_face_landmarks.dat')

    def loadSingleImgLoad(self,imgPath):
        # Load an image of a person
        return DataLoader.load_image(imgPath)

    def getBoundedBox(self,jc_orig):
        # Detect face and return bounding box
        return self.alignment.getLargestFaceBoundingBox(jc_orig)

    def cropImage(self,size, jc_orig, bb):
        # Transform image using specified face landmark indices and crop image to 96x96
        return self.alignment.align(size, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
class Displayer():
    def __init__(self):
        pass
    def showImage(self, jc_orig, size=131):
        # Show original image
        plt.subplot(size)
        plt.imshow(jc_orig)
        return self

    def showImageWithBB(self,jc_orig,bb,size=131):
        # Show original image with bounding box
        plt.subplot(size)
        plt.imshow(jc_orig)
        plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
        return self

    def showAlignedImage(self,jc_aligned,size):
        # Show aligned image
        plt.subplot(size)
        plt.imshow(jc_aligned)
        return self

    def showAccuracy(self, distances, identical):
        distances = np.array(distances)
        identical = np.array(identical)

        thresholds = np.arange(0.02, 0.4, 0.005)

        f1_scores = [f1_score(identical, distances < t) for t in thresholds]
        acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

        opt_idx = np.argmax(f1_scores)
        # Threshold at maximal F1 score
        opt_tau = thresholds[opt_idx]
        # Accuracy at maximal F1 score
        opt_acc = accuracy_score(identical, distances < opt_tau)

        # Plot F1 score and accuracy as function of distance threshold
        plt.plot(thresholds, f1_scores, label='F1 score');
        plt.plot(thresholds, acc_scores, label='Accuracy');
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
        plt.xlabel('Distance threshold')
        plt.legend()
        plt.show()

        dist_pos = distances[identical == 1]
        dist_neg = distances[identical == 0]

        plt.figure(figsize=(12,4))

        plt.subplot(121)
        plt.hist(dist_pos)
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title('Distances (pos. pairs)')
        plt.legend();

        plt.subplot(122)
        plt.hist(dist_neg)
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title('Distances (neg. pairs)')
        plt.legend()
        plt.show()
        return self

    def showScatter(self, embedded):
        X_embedded = TSNE(n_components=2).fit_transform(embedded)

        for i, t in enumerate(set(Trainer.targets)):
            idx = Trainer.targets == t
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)   

        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()


    def run(self):
        plt.show()
class Trainer():
    def __init__(self, dataloader : DataLoader):
        self.nn4_small2_pretrained = create_model()
        self.nn4_small2_pretrained.load_weights('modules/dependencies/weights/nn4.small2.v1.h5')
        self.distances = []
        self.identical = []
        self.metadata = dataloader.load_metadata(path='./pictures')
        self.dtloader = dataloader
        self.svc = LinearSVC()
        self.knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        # Initialize the OpenFace face alignment utility
        self.alignment = AlignDlib('modules/dependencies/models/shape_predictor_68_face_landmarks.dat')

    def align_image(self,img):
        return self.alignment.align(96, img, self.alignment.getLargestFaceBoundingBox(img), 
                            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    def preProcess(self):
        embedded = np.zeros((self.metadata.shape[0], 128))
        TTD_size = self.metadata.shape[0]


        dem = 0

        for i, m in enumerate(self.metadata):
            print(m.image_path())
            img = self.dtloader.load_image(m.image_path())
            img = self.align_image(img)
            if img is not None:
                # scale RGB values to interval [0,1]
                img = (img / 255.).astype(np.float32)
                # obtain embedding vector for image
                embedded[i] = self.nn4_small2_pretrained.predict(
                    np.expand_dims(img, axis=0))[0]
                dem = dem + 1


        if dem < TTD_size:
            while True:
                mm, nn = embedded.shape
                flagThoat = True
                for i in range(0, mm):
                    if np.sum(embedded[i]) == 0:
                        embedded = np.delete(embedded, i, 0)
                        self.metadata = np.delete(self.metadata, i, 0)
                        flagThoat = False
                        break
                if flagThoat == True:
                    break
        return embedded
    def distance(self,emb1, emb2):
        return np.sum(np.square(emb1 - emb2))
    def train(self,embedded):
        
        num = len(self.metadata)
        for i in range(num - 1):
            for j in range(i + 1, num):
                self.distances.append(self.distance(embedded[i], embedded[j]))
                self.identical.append(1 if self.metadata[i].name == self.metadata[j].name else 0)
                
        

        targets = np.array([m.name for m in self.metadata])

        encoder = LabelEncoder()
        encoder.fit(targets)

        # Numerical encoding of identities
        y = encoder.transform(targets)

        train_idx = np.arange(self.metadata.shape[0]) % 2 != 0
        test_idx = np.arange(self.metadata.shape[0]) % 2 == 0

        # 50 train examples of 10 identities (5 examples each)
        X_train = embedded[train_idx]
        # 50 test examples of 10 identities (5 examples each)
        X_test = embedded[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        self.knn.fit(X_train, y_train)
        self.svc.fit(X_train, y_train)

        acc_knn = accuracy_score(y_test, self.knn.predict(X_test))
        acc_svc = accuracy_score(y_test, self.svc.predict(X_test))

        print(acc_knn)
        print(acc_svc)
    def save(self):
        joblib.dump(self.svc,'svc.pkl')
        joblib.dump(self.knn,'knn.pkl')
class Tester():
    def __init__(self,dataloader : DataLoader,trainer : Trainer,loader : Loader):
        self.dtloader = dataloader
        self.trainer = trainer
        self.loader = loader
    
    def indentifyImage(self,imgUrl : str, mydict : Array):
        # Tran Tien Duc them vao
        filename_test = imgUrl
        img_test = self.dtloader.load_image(filename_test)
        img_test = self.trainer.align_image(img_test)
        # scale RGB values to interval [0,1]
        img_test = (img_test / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedded_test = self.loader.nn4_small2_pretrained.predict(np.expand_dims(img_test, axis=0))[0]

        test_prediction = self.trainer.svc.predict([embedded_test])

        result = mydict[test_prediction[0]]
        print(result)

        plt.imshow(img_test)
        plt.title(f'Recognized as {result}')
        plt.show()

    def loadModel(self,path):
        self.trainer.svc = joblib.load(path)
class Controller():
    """
    self.dtloader : DataLoader()
    """
    def __init__(self) -> None:
        self.dtloader = DataLoader()
        self.loader = Loader(self.dtloader)
        self.displayer = Displayer()
        self.trainer = Trainer(self.dtloader)
        self.tester = Tester(self.dtloader, self.trainer, self.loader)

    def getDtLoader(self):
        return self.dtloader

    def getLoader(self):
        return self.loader

    def getDisplayer(self):
        return self.displayer

    def getTrainer(self):
        return self.trainer

    def getTester(self):
        return self.tester
