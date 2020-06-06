import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
mod_dir = this_dir.replace('tests','')
sys.path.append(mod_dir)
