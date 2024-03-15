#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

this code is designed to show the reproducibility of the results
if you choose instance VAN s 12 1, you would have the result for the experiments of the hybrid approach for T=12
intitialized by the result of the deep learning algorithm (CNN+LSTM) 
; the result for VAN s 24 1, and VAN s 48 1
is related to scaling part


***********
you could run the function at the end of the descript "hybrid_algorithm_solution" found in line 621
*********

"""

from instance import Instance
from datetime import datetime

import csv

from hydraulics import HydraulicNetwork
from pathlib import Path
from stats import Stat
import os


import datetime as dt

from ast import literal_eval

from networkanalysis import NetworkAnalysis
from new_partition import NetworkPartition

import time
import configgenerator_me_coupling
import second_subproblem
import numpy as np

import random
import math

import pickle


OA_GAP = 1e-2
MIP_GAP = 1e-6

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'D0': 1, 'H0': '/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'D0': 21, 'H0': '/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'D0': 1, 'H0': '/01/2013 00:00'},
    'VAN': {'ntk': 'Vanzyl', 'D0': 21, 'H0': '/05/2011 08:00'}
}
PROFILE = {'s': 'demand_tank0_2011_lazy_bum_van_gaus', 'n': 'demand_tank0_2011_lazy_bum_van_gaus'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}


# ex of instance id: "FSD s 24 3"
def makeinstance(instid: str):
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    dbeg = f"{(d['D0'] + int(a[3]) - 1):02d}" + d['H0']
    dend = f"{(d['D0'] + int(a[3])):02d}" + d['H0']
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])


def makeinstance_mein_loop(instid: str, ite):
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    dbeg= dt.datetime.strftime(dt.datetime.strptime('01/01/2011 08:00','%d/%m/%Y %H:%M')+(ite)*dt.timedelta(hours=24),'%d/%m/%Y %H:%M')
    dend= dt.datetime.strftime(dt.datetime.strptime('01/01/2011 08:00','%d/%m/%Y %H:%M')+(ite+1)*dt.timedelta(hours=24),'%d/%m/%Y %H:%M')
#    dbeg = f"{(d['D0'] + int(a[3]) - 1):02d}" + d['H0']
#    dend = f"{(d['D0'] + int(a[3])):02d}" + d['H0']
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])




def parsemode(modes):
    pm = {k: mk[0] for k, mk in MODES.items()}
    if modes is None:
        return pm
    elif type(modes) is str:
        modes = [modes]
    for k, mk in MODES.items():
        for mode in mk:
            if mode in modes:
                pm[k] = mode
                break
    return pm







TESTNETANAL = True

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'H0': '01/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'H0': '21/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'H0': '01/01/2013 00:00'},
    'RIY': {'ntk': 'Richmond', 'H0': '01/01/2012 00:00'},
    'VAN': {'ntk': 'Vanzyl', 'H0': '21/05/2011 08:00'},
}
PROFILE = {'s': 'demand_tank0_2011_lazy_bum_van_gaus', 'n': 'demand_tank0_2011_lazy_bum_van_gaus', 'y': 'demand_tank0_2011_lazy_bum_van_gaus'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}
PARAMS = {

    'ANY s 48': {'mipgap': 1e-6, 'vdisc': 20, 'safety': 10},
    'RIY y 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'VAN s 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'default' : {'mipgap': 1e-6, 'vdisc': 10, 'safety': 2}}
FASTBENCH = [
    'FSD s 12 1',
    'FSD s 24 1',

]
OUTDIR = Path("../output/")
OUTFILE = Path(OUTDIR, f'resallex.csv')
HEIGHTFILE = Path("../data/Richmond/hauteurs220222.csv")
""" solution mode: 'EXIP' (default: IP extended model), 'EXLP' (LP extended relaxation)
    time adjustment heuristic: NOADJUST (default: no heuristic) """
MODES = {"solve": ['EXIP', 'EXLP'],
         "adjust": ['NOADJUST']}


def defaultparam(instid: str) -> dict:
    """ return the default parameter values for the given instance. """
    params = PARAMS.get(instid[:8])
    return params if params else PARAMS['default']


def parsemode(modes: str) -> dict:
    """ read the exec mode (see MODES with space separator, e.g.: 'EXIP NOADJUST' ). """
    pm = {k: mk[0] for k, mk in MODES.items()}
    if modes is None:
        return pm
    ms = modes.split()
    for k, mk in MODES.items():
        for mode in mk:
            if mode in ms:
                pm[k] = mode
            break
    return pm


def makeinstance(instid: str) -> Instance:
    """ create the instance object named instid, e.g.: "FSD s 24 3". """
    datefmt = "%d/%m/%Y %H:%M"
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    day = int(a[3]) - 1
    assert day in range(0, 366)
    dateday = dt.datetime.strptime(d['H0'], datefmt) + dt.timedelta(days=day)
    dbeg = dateday.strftime(datefmt)
    dateday += dt.timedelta(days=1)
    dend = dateday.strftime(datefmt)
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])


def parseheigthfile(instid: str, solfilepath: Path) -> dict:
    """
    parse the solution file including height profiles and return the solution for just one instance is specified;
    return {instid: {'inactiveplan': {t: set(inactive arcs at t), forall time t},
            'dhprofiles': {tk: [dh_tk[t] forall time t] forall tank tk}}}.
    """
    csvfile = open(solfilepath)
    # instid = instid.replace(" y ", " s ") if instid and instid.startswith("RIY y 24") else None
    rows = csv.reader(csvfile, delimiter=';')
    solutions = {}
    for row in rows:
        if (not instid) or (row[0].strip() == instid):
            strdict = literal_eval(row[2].strip())
            solutionplan = {(k[1], k[2]): v for k, v in strdict.items() if len(k) == 3 and k[0] == 'X'}
            dhprofiles = {k[1]: v for k, v in strdict.items() if len(k) == 2 and k[0] == 'DH'}
            inactiveplan = {t: set(k for k, v in solutionplan.items() if v[t] == 0) for t in range(24)}
            solutions[row[0].strip()] = {'inactiveplan': inactiveplan, 'dhprofiles': dhprofiles}
    return solutions







def second_subprob(instance: Instance, q_inf, penalt, params: dict, stat: Stat, drawsolution: bool, meanvolprofiles: list = None):
    if meanvolprofiles is None:
        meanvolprofiles = []
    print(f'********** SOLVE EXTENDED MODEL ************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())


    gentime = time.time()


    model = second_subproblem.build_model(instance, q_inf, penalt)
    # model.write('extendedmodel.lp')
    model.params.MIPGap = params["mipgap"]
    model.params.timeLimit = 3600
    # model.params.OutputFlag = 0
    # model.params.Threads = 1
    # model.params.FeasibilityTol = 1e-5
    model.optimize()
    model_lp= model.getObjective()
    obj_lp=model_lp.getValue()

    
    k_h_=[]
    for i in range(0,len(model.getVars())):

            k_h_.append([model.VarName[i], model.X[i]])

            k_h_arr=np.array(k_h_)
            keydicts= k_h_arr[:, 0]
            bt_h_p= dict([
                (key, [float(k_h_arr[i][1])]) for key, i in zip(keydicts, range(len(k_h_arr)))])
    level={}
    inf_l={}
            
    for ts in range(0, len(instance.horizon())):
        for k, tank in instance.tanks.items():
#            level[k, ts]= bt_h_p[f'ht({k},{ts})']
            inf_l[k, ts]= bt_h_p[f'qr({k},{ts})'][0]
            
    for ts in range(0, len(instance.horizon())+1):
        for k, tank in instance.tanks.items():
            level[k, ts]= bt_h_p[f'ht({k},{ts})'][0]
    end_time= time.time()-gentime
            
    model.terminate()
    return level, inf_l, obj_lp, end_time
    






def first_subproblem(instance: Instance, levels, penalt, params: dict, stat: Stat, drawsolution: bool, meanvolprofiles: list = None):
    """ generate and solve the extended pump scheduling model (either LP or ILP according to the mode)"""

    feastol = params["mipgap"]
    network = NetworkAnalysis(instance, feastol) if TESTNETANAL \
        else HydraulicNetwork(instance, feastol=feastol)
    netpart= NetworkPartition(instance)

    print("generate configurations")
    gentime = time.time()
    col={}
    columns = configgenerator_me_coupling.ConfigGen(instance, network, netpart, levels, penalt, feastol, params["vdisc"], params["safety"],
                                        meanvolprofiles)
####    print("yuck")
####    print(columns)
    
    

    for t in instance.horizon():
        col[t]= columns.generate_all_columns_me(t)[t]
        inde=min(col[t])
    new_dict = {}
    for t, inner_dict in col.items():
        new_dict[t] = {}
        for (c, k), value in inner_dict.items():
            if k in new_dict[t]:
                new_dict[t][k].append([c, value])
            else:
                new_dict[t][k] = [[c, value]]
                

    min_value_dic={}
    min_value_nest= {}
    for t in instance.horizon():
        min_value_nest[t]={}
        for compo in new_dict[t].keys():
            min_tempp_value=10000000
            for kj, kn in new_dict[t][compo]: 
                if kn['power']<= min_tempp_value:
                    min_tempp_value = kn['power']
                    min_value_dic[t,compo]= [kj, kn]
                    min_value_nest[t][compo]=[kj,kn]
                else:
                    pass

    cpugen = time.time() - gentime
    return cpugen, min_value_nest




def load_and_process_data(outit, instance, it, h_lev):
    if outit == 0:
        cc = np.load('mean_50_test.npy')
    else:
        cc = np.load('several_50_test.npy')[outit]
    
    cc = cc.reshape(50, 12, 2)

    for k in range(len(cc[1])):
        if len(instance.horizon())==12:
            for tt, tank in instance.tanks.items():
                zz = 0 if tt == 't6' else 1 if tt == 't5' else None
                if zz is not None:
                    if k == 0:
                        h_lev[tt, 0] = tank.head(tank.vinit)
                    h_lev[tt, k+1] = cc[it][k][zz]
        elif len(instance.horizon())==24:
            for tt, tank in instance.tanks.items():
                zz = 0 if tt == 't6' else 1 if tt == 't5' else None
                if zz is not None:
                    if k == 0:
                        h_lev[tt, 0] = tank.head(tank.vinit)
                        h_lev[tt, 2*k+1]= (1/2)*(cc[it][k][zz]+tank.head(tank.vinit))
                        h_lev[tt, 2*k+2]= cc[it][k][zz]
                    else:
                        h_lev[tt, 2*k+1]= (1/2)*(cc[it][k-1][zz]+cc[it][k][zz])
                        h_lev[tt, 2*k+2]= cc[it][k][zz]
        elif len(instance.horizon())==48:
            for tt, tank in instance.tanks.items():
                zz = 0 if tt == 't6' else 1 if tt == 't5' else None
                if zz is not None:
                            if k==0 :
                                h_lev[tt, 0]= tank.head(tank.vinit)
                                h_lev[tt, 4*k+1]= (1/4)*(cc[it][k][zz])+(3/4)*(tank.head(tank.vinit))
                                h_lev[tt, 4*k+2]= (1/2)*(cc[it][k][zz])+(1/2)*(tank.head(tank.vinit))
                                h_lev[tt, 4*k+3]= (3/4)*(cc[it][k][zz])+(1/4)*(tank.head(tank.vinit))
                                h_lev[tt, 4*k+4]= (cc[it][k][zz])
                                
                            else:
                                h_lev[tt, 4*k+1]= (3/4)*(cc[it][k-1][zz])+(1/4)*cc[it][k][zz]
                                h_lev[tt, 4*k+2]= (2/4)*(cc[it][k-1][zz])+(2/4)*cc[it][k][zz]
                                h_lev[tt, 4*k+3]= (1/4)*(cc[it][k-1][zz])+(3/4)*cc[it][k][zz]
                                h_lev[tt, 4*k+4]= (cc[it][k][zz])
            
            
    
    return {k: v for k, v in h_lev.items()}



def solveinstance_second_coupling(instid: str, starting_instance, final_instance, same_trajectory, params: dict = None, modes: str = "", stat: Stat = None, drawsolution: bool = True,
                  outfile: Path = OUTFILE):
    """ solve the extended model for a given instance: report the result in 'outfile' """
    if params is None:
        params = defaultparam(instid)

    
    
    levels, penalt, q_inf, big_flag, diff_dict, diff_h1, Levius, tank_in_out, configggg, numb_ite, diff_hh, mattt, powerr = (dict() for _ in range(13))
    inflo, counttt = 0, 0

    

    
    
    #the test days are from 0 to 50; it shows the number of instances to consider
    for it in range(starting_instance, final_instance):
        
        instance = makeinstance_mein_loop(instid, it)
        stat = Stat(parsemode(modes)) if stat is None else stat
        solution = parseheigthfile(instid, HEIGHTFILE).get(instid)
        dhprofiles = solution.get('dhprofiles') if solution else None
        meanvolprofiles = instance.getvolumeprofiles(dhprofiles)
        
        
        #only for the T=12 can be true
        if same_trajectory== True:
            horiz=len(instance.horizon())
            
            #loading same update penalty
            with open(f'penalty_reported_50_test_{horiz}.pickle', 'rb') as f:

                penalty_pad = pickle.load(f)

        
        h_lev, q_inff = {}, {}
        dictionaries_to_initialize = [penalt, Levius, tank_in_out, mattt, diff_h1, diff_hh, diff_dict, powerr, numb_ite]
        for d in dictionaries_to_initialize:
            d[it] = {}

        for outit in range(40):
            penalt[it][outit] = {iteration: {} for iteration in range(1, 52)}
            q_inff = {iteration-1: {} for iteration in range(1, 52)}




        iteration, outit = 0, 0
        dictionaries_to_initialize = [big_flag, Levius, tank_in_out, penalt]
        for d in dictionaries_to_initialize:
            d[it] = {}

        penalt[it][outit] = {iteration: {}}


        for outit in range(0, 35):
            
            penalt[it][outit]={}
            penalt[it][outit][0]={}
            
            for t in range(0, len(instance.horizon())):
                for k, tank in instance.tanks.items():
                    penalt[it][outit][0][k, t]= 50

    
            levels = load_and_process_data(outit, instance, it, h_lev)
                                

            
            for t in instance.horizon():
                for k, tank in instance.tanks.items():
                    q_inf[k, t]= (levels[k, t+1]-levels[k, t])/instance.flowtoheight(tank)



            

            dictionaries_to_initialize = [Levius, tank_in_out, big_flag, numb_ite, mattt, diff_h1, diff_hh, diff_dict, powerr]
            for d in dictionaries_to_initialize:
                d.setdefault(it, {})[outit] = {}

            for iteration in range (0, 5):
            
                dictionaries_to_initialize = [Levius, tank_in_out, numb_ite, mattt, diff_h1, diff_hh, diff_dict, powerr]
                for d in dictionaries_to_initialize:
                    d.setdefault(it, {}).setdefault(outit, {})[iteration] = {}

                ite, stopping_cri, sm_h = 0, 10000, 1000

            
                for i in range (0, 85):
                
                
                    dictionaries_to_initialize = [diff_h1, diff_hh, diff_dict, Levius, tank_in_out]
                    for d in dictionaries_to_initialize:
                        d.setdefault(it, {}).setdefault(outit, {}).setdefault(iteration, {})[i] = {}

                    if stopping_cri<1e-3 or sm_h <1e-3:
                        break
                
            
                    stopping_cri_dic={}
                    
                    
                    #first sub-problem solved by enumearion (inputs are levels from second subproblem(or initialization) and penalties)----outputs are the configuration and the time require to solve it
                    timetoget, min_value_nest = first_subproblem(instance, levels, penalt[it][outit][iteration], params, stat, drawsolution, meanvolprofiles)
                    
                    
                    #the tank in-outflow is resulted from first subproblem 
                    q_inf = {(k, t): min_value_nest[t][cmp][1]['mein_tank'][k] for t in instance.horizon() for cmp, rmd in min_value_nest[t].items() for k, tank in instance.tanks.items()}
                    

                    #second subproblem an LP inputs are the tank inflow and outflow from first subproblem and the penalty term ---- outputs are the levels the inflow outflow of the tanks from second subproblwm and the time requires to solve it
                    levels, inflo, objec, time_sec = second_subprob(instance, q_inf, penalt[it][outit][iteration], params, stat, drawsolution, meanvolprofiles)
                
                
                    Levius[it][outit][iteration][i]= levels
                    tank_in_out[it][outit][iteration][i]= inflo
                    mattt[it][outit][iteration][i]= min_value_nest
                    
                    
                    
                    #####here computing the stopping criteria and different distance between two solutions derived from first and second subproblem#######
                
                    for t in instance.horizon():
                        for k, tank in instance.tanks.items():                        
                            q_inff[ite][k, t]= q_inf[k, t]
                            if ite>=1:
                                stopping_cri_dic[k, t]= abs(q_inff[ite][k, t]-q_inff[ite-1][k, t])
                    if ite>=1:
                        stopping_cri=max((stopping_cri_dic.values()))
                        
                    ite+=1
                            
                    numb_ite[it][outit][iteration][i]=[i, timetoget+time_sec, stopping_cri]
                    
                    diff_dict[it][outit][iteration][i] = {k: [[v[1]['mein_tank'][ln] - inflo[ln, k] for ln, lm in v[1]['mein_tank'].items()] for kk, v in val.items()] for k, val in min_value_nest.items()}
             
                    diff_h1[it][outit][iteration][i] = {k: [[v[1]['tank_level'][ln] - levels[ln, k+1] for ln, lm in v[1]['tank_level'].items()] for kk, v in val.items()] for k, val in min_value_nest.items()}

                    diff_hh[it][outit][iteration][i] = {(ln, k): v[1]['tank_level'][ln] - levels[ln, k+1] for k, val in min_value_nest.items() for kk, v in val.items() for ln, lm in v[1]['tank_level'].items()}


                    
    
                       
                    sm_h= sum(sum(abs(x) for x in klmmn) for ii, jj in diff_h1[it][outit][iteration][i].items() for klmmn in diff_h1[it][outit][iteration][i][ii])
                
                    powerr[it][outit][iteration][i] = sum(min_value_nest[k][mg][1]['power'] for k, m in min_value_nest.items() for mg, mk in m.items())
                
                

            
                big_flag[it][outit][iteration] = sum(sum(klmmn) for ii, jj in diff_dict[it][outit][iteration][ite-1].items() for klmmn in jj)

            
                configggg[it] = min_value_nest
                
                #if the solution is feasible(no distance between solution of first and second declare feasibility and count it)
                if abs(big_flag[it][outit][iteration]) <=1e-6:
                    counttt+=1
                    print("******solved******")
                    break
            
                

                            
                bbb=list(diff_hh[it][outit][iteration].keys())[-1]-1
                
                
                penalt[it][outit][iteration+1]={}
                
                if same_trajectory==True:
            
                    for ii, jj in diff_hh[it][outit][iteration][bbb].items():
                        penalt[it][outit][iteration+1][ii]= penalty_pad[it][outit][iteration+1][ii]
                else:
                    
                    for ii, jj in diff_hh[it][outit][iteration][bbb].items():
                        if abs(diff_hh[it][outit][iteration][bbb][ii])>=1e-3:
                            penalt[it][outit][iteration+1][ii]= 5*random.uniform(0.75,1)*math.exp(-iteration/10)* penalt[it][outit][iteration][ii]+1                  
                        else:
                            penalt[it][outit][iteration+1][ii]= 2*random.uniform(0.75,1)*math.exp(-iteration/10)* penalt[it][outit][iteration][ii]+1

                    
            if abs(big_flag[it][outit][iteration]) <=1e-6:
                break
            

    
    return mattt, tank_in_out, timetoget, ite, stopping_cri, numb_ite, diff_dict, big_flag, Levius, penalt, counttt, stopping_cri_dic, powerr, configggg, objec, diff_h1, diff_hh





def hybrid_algorithm_solution(instid, starting_instance, final_instance, same_trajectory, params: dict = None, modes: str = "", stat: Stat = None, drawsolution: bool = True,
                  outfile: Path = OUTFILE):
    
    
    mat1, mat2, mat3, mat4, stopping_cri, numb, diff_d, big_flag, leve, penalization, shomar, stop_di, power, confg, objec_v, diff_scheiss, diff_hh= solveinstance_second_coupling(instid, starting_instance, final_instance, same_trajectory, modes='EXIP')
    
    
    cost={}
    for inst, inst_val in power.items():
            out_it_list=[]
            for out_it, outit_val in inst_val.items():
                out_it_list.append(out_it)
                inner_loop_list=[]
                for inner_ke, inner_val in inst_val[out_it].items():
                    inner_loop_list.append(inner_ke)
                    in_inner_list=[]
                    lis_inner_val= list(inner_val.values())
                    cost[inst]=lis_inner_val[-1]
    t_tim_dic={}
    t_tim_end={}
    count_out=0
    for ii in numb.keys():
        t_sep=0
        t_tim_dic[ii]={}
        t_tim_end[ii]={}
        for keyy_outit, valuuu in numb[ii].items():
          count_out=count_out+1
          for key_iterat, val_in in valuuu.items():
                  for key_in, val_nn in val_in.items():
                      t_sep= t_sep+ val_nn[1]
        t_tim_end[ii]=t_sep
        
    convergence_gap={}
    for inst, inst_val in big_flag.items():
            out_it_list=[]
            for out_it, outit_val in inst_val.items():
                out_it_list.append(out_it)
            last_one= list(big_flag[inst][out_it_list[-1]].values())
            convergence_gap[inst]=last_one[-1]
            
    return cost, t_tim_end, convergence_gap, shomar


#####starting and final_instance can be anything from 0 to 50
starting_instance=0
final_instance=2
numb_inst=final_instance-starting_instance

###for Van12 we have saved the penalty of the reported values, for 24 and 48 we don't have penalty values; however, the performance over instances must be the same
##in case of T=12 it can True
same_trajectory=False

####three options: VAN s 12 1 indicating T=12, but also possible to check VAN s 24 1 for T=24 and VAN s 48 1 for T=48
instid= 'VAN s 12 1'

#instid= 'VAN s 12 1'
#instid= 'VAN s 24 1'
#instid= 'VAN s 48 1'

cost, time_require, feasibility_measure, solved_problems= hybrid_algorithm_solution(instid, starting_instance, final_instance, same_trajectory ,  modes='EXIP')

print(f"number of solved instances {solved_problems} among {numb_inst}")
print(f"feasibility measures {feasibility_measure}")
