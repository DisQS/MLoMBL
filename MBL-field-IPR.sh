#!/bin/bash

# settings from input

size=${1:-10}
seed=${2:-1}
config=${3:-2}
keep=${4:-1}

echo "MBL-F: making for L=" $size "with starting seed=" $seed "and" $config "samples"

# settings for directories

currdir=`pwd`

for bJ in `seq 0.1 0.1 4.0`
do

echo "--- bJ=" $bJ

jobdir="$size-IPR"
mkdir -p $jobdir

jobname=`printf "$jobdir-bJ$bJ"`
echo $jobname

jobfile=`printf "$jobname.sh"`
logfile=`printf "$jobname.log"`
pythonfile=`printf "$jobname.py"`

# settings for parallel submission

cd $jobdir

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=3700
#SBATCH --account=su007-rr

module purge
module restore sul2
module load QuSpin/0.3.6

MY_PARALLEL_OPTS="-N 1 --delay .2 -j \$SLURM_NTASKS --joblog $jobname-parallel-\${SLURM_JOBID}.log"
MY_SRUN_OPTS="-N 1 -n 1 --exclusive"
MY_EXEC="$currdir/$jobdir/$pythonfile"
chmod +x ${pythonfile}
parallel \$MY_PARALLEL_OPTS srun \$MY_SRUN_OPTS \$MY_EXEC ::: {1..$config}

EOD

cat > ${pythonfile} << EOD 
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
import time

import quspin
from quspin.operators import hamiltonian,quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time, diag_ensemble # t_dep measurements
from quspin.tools.Floquet import Floquet, Floquet_t_vec # Floquet Hamiltonian

#set seeds
JJ   =1

#left=\$bJ
#right=\$bJ
#step=10
#blist=np.linspace(left,right,step)
blist=[$bJ]
#small=4
#big=11
#stepsize1=1
#Llist=np.arange(small,big,stepsize1)
Llist=[$size]
it=1
beta=0.721
a1=12345*$seed
a2=23456*$seed
a3=34567*$seed
a4=45678*$seed
batch=10
#path='/gpfs/home/p/phrczh/Qusp/MBLfield'
#path='/gpfs$currdir/$jobdir'
path='$currdir/$jobdir'
#print(path)
#blist=blist.tolist()
#Llist=Llist.tolist()

for L in Llist:
    #t=[]
    #t1=time.time()
    for b in blist:
        #print("python:",sys.argv[1])
        pre=a1+a2*L+int(a3*b)+a4*((int(sys.argv[1])))
        np.random.seed(pre)
        see=np.random.randint(pre)
        np.random.seed(see)
        basis=spin_basis_1d(L=L,pauli=False,Nup=L//2)
        #print(basis)
        J_nn=[[JJ,i,(i+1)%L] for i in range(L)] # PBC
        h_z=[[2*b*np.random.random()-b,i] for i in range(L)]
        # static and dynamic lists
        static=[["zz",J_nn],["xx",J_nn],["yy",J_nn]]
        field=[["z",h_z]]
        #print(static[0][1][0])
        # compute Hamiltonians
        H=hamiltonian(static,[],dtype=np.float64,basis=basis)+hamiltonian(field,[],dtype=np.float64,basis=basis)
        #print(H)
        E_fermion,V=H.eigh()
        IPR=np.zeros(len(E_fermion))
        for i in range (len(IPR)):
            IPR[i]=sum(abs(V[j,i])**4 for j in range (len(E_fermion)))
        #print(IPR)
        outfile=f"/L={L}-{(int(sys.argv[1]))}-{'%.1f' %b}.csv"
        #print(path)
        #print(outfile)
        np.savetxt(path+outfile,np.c_[E_fermion,IPR])
    #t2=time.time()
    #t.append(t2-t1)
    #np.savetxt(os.path.join(path,f"{L}/L={L}-{1+(int(sys.argv[1]))}-time.csv"),t)


print("-- finished")

EOD

chmod 755 ${jobfile}
chmod 755 ${pythonfile}
#(msub -q devel $jobdir/${jobfile}) # for queueing system
#(sbatch -q devel $jobdir/${jobfile}) # for queueing system

sbatch ${jobfile} # for queueing system
#sbatch -p devel ${jobfile} # for queueing system

#(source $jobdir/${jobfile} ) >& $jobdir/${logfile} & # for parallel shell execution
#source ${jobfile} #>& ${logfile} # for sequential shell execution

sleep 1

cd ..

done
