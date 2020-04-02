from joblib import Memory
import os

disk = Memory(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache'), verbose=0)
