import torch
import torchrl
import benchmarl
import pettingzoo

def test_imports():
    assert torch.__version__ is not None
    assert benchmarl.__version__ is not None
    print("Smoke test passed: All dependencies resolve perfectly!")
