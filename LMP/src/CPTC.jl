module CPTC

import LatticeQCD
import Gaugefields
import LatticeDiracOperators

import Wilsonloop



using LinearAlgebra

include("elementry_base.jl")  # ✅ 로컬 패키지를 가져옴
include("Layer_sequence.jl") # ✅ 로컬 패키지를 가져옴
using .elementry_base: Paths, Onestep, Hopping_operator, PTC, Random_SpinM, Spin_matrix, Generate_weight, CPTC

using .Layer_sequence: Lossfunction, PTC_Layer, extend_layer











export Paths
export Onestep
export Hopping_operator
export PTC
export Random_SpinM
export Spin_matrix
export Generate_weight
export CPTC



#-----------------------------------------------------------------------------

export Lossfunction
export PTC_Layer
export extend_layer



#-----------------------------------------------------------------------------




end
