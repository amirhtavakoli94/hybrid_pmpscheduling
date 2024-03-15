#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solving the second subproblem of the hybrid algorithm
"""

import gurobipy as gp
from gurobipy import GRB

from instance import Instance



# !!! check round values
# noinspection PyArgumentList
##def build_model(inst: Instance, q_inf, penalt, oagap: float, arcvals=None):
def build_model(inst: Instance, q_inf, penalt, arcvals=None):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')
    milp.params.outputflag = 0


    hvar = {}  # node head
    qexpr = {}  # node inflow
    
    epsi = {}

    nperiods = inst.nperiods()
    horizon = inst.horizon()

    for t in horizon:
#        for (i, j), pump in inst.pumps.items():
#            ivar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'ik({i},{j},{t})')

        for j in inst.junctions:
            hvar[j, t] = milp.addVar(name=f'hj({j},{t})')

        for j, res in inst.reservoirs.items():
            hvar[j, t] = milp.addVar(lb=res.head(t), ub=res.head(t), name=f'hr({j},{t})')

        for j, tank in inst.tanks.items():
            lbt = tank.head(tank.vinit) if t == 0 else tank.head(tank.vmin)
            ubt = tank.head(tank.vinit) if t == 0 else tank.head(tank.vmax)
            hvar[j, t] = milp.addVar(lb=lbt, ub=ubt, name=f'ht({j},{t})')

        milp.update()



    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},{nperiods})')

    milp.update()

    # FLOW CONSERVATION
    for t in horizon:
        for j, tank in inst.tanks.items():
            qexpr[j, t] = milp.addVar(lb=-1000,ub=1000, name=f'qr({j},{t})')

        for j, tank in inst.tanks.items():
            milp.addConstr(hvar[j, t+1] - hvar[j, t] == inst.flowtoheight(tank) * qexpr[j, t], name=f'fc({j},{t})')

    for t in horizon:
        for j, tank in inst.tanks.items():
            
            epsi[j, t]=milp.addVar(lb=0)
            
            milp.addConstr(epsi[j, t]>= qexpr[j, t]-q_inf[j, t])
            milp.addConstr(epsi[j, t]>= -qexpr[j, t]+q_inf[j, t])
            
    milp.update()


    obj= gp.quicksum(penalt[k, t]*(epsi[k, t])
                      for k, tank in inst.tanks.items() for t in horizon)

    milp.setObjective(obj, GRB.MINIMIZE)
    milp.update()


    milp._hvar = hvar
    milp._obj = obj

    return milp




