import pytest
import numpy as np
from sharc.propagation.propagation_p2001 import PropagationP2001
from sharc.parameters.parameters_p2001 import ParametersP2001

def test_p2001_instantiation():
    """Teste básico apenas para garantir que a classe é carregada sem erros"""
    # 1. Instancia o modelo com os argumentos necessários
    rng = np.random.RandomState(42)
    params = ParametersP2001()
    modelo_p2001 = PropagationP2001(rng, params)
    
    # Se chegou até aqui sem dar erro, o import e os mapas estão funcionando!
    assert modelo_p2001 is not None