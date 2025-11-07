#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 18:22:27 2025

@author: usuario
"""

import numpy as np
import matplotlib.pyplot as plt

from pytc2.sistemas_lineales import plot_plantilla
from pytc2.filtros_digitales import fir_design_pm
import pytc2
print(pytc2.__file__)


# --- CONFIGURACIÓN INICIAL ---
plt.close('all')

# --- EJEMPLO DE USO DE LA PLANTILLA ---
# Graficar plantilla estándar para filtros
plot_plantilla()