from settings import MainPathData
from utils import UrbanPlanningGAN
import sys


UPGAN = UrbanPlanningGAN()
UPGAN.load_data()
UPGAN.train(n_epochs=int(sys.argv[1]), n_batch=int(sys.argv[2]))
UPGAN.d_model.save_weights(MainPathData.data.base.joinpath('d_model.h5'))
UPGAN.g_model.save_weights(MainPathData.data.base.joinpath('g_model.h5'))
UPGAN.gan_model.save_weights(MainPathData.data.base.joinpath('gan_model.h5'))
