import numpy as np;
import math
import sys

########################################################################
########################################################################

# loops thru range of alphas for equities to maximize return on volatility
# this is the code used in the posted paper
########################################################################
########################################################################


########################################################################
#inputs for years: 1889-1970
########################################################################



year=1889;
mu=0.0183;
delta=0.0357
sigma = -0.14
bond_target=0.008;
stock_target=0.0698;
pie_target=0.5; 
verbose=0;
alpha_min=0; alpha_max=15;

np.set_printoptions(precision=4)






########################################################################
# get options
########################################################################

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]


for i in range (0,len(opts)):
	if opts[i]=="-pt": pie_target=float(args[i]);	## stationary propability
	if opts[i]=="-mu": mu=float(args[i]);		## mean consumption	
	if opts[i]=="-delta": delta=float(args[i]);	## standard deviation of consumption	
	if opts[i]=="-sigma": sigma=float(args[i]);	## serial correlation of consumption	
	if opts[i]=="-st": stock_target=float(args[i]);	## mean return of stocks
	if opts[i]=="-bt": bond_target=float(args[i]);	## mean return of riskless bonds	
	if opts[i]=="-amax": alpha_max=int(args[i]);	## max value of alpha	
	if opts[i]=="-v": verbose=int(args[i]);		## verbose flag
	if opts[i]=="-year": year=int(args[i]);		## start year for preset data	


print("start year",year);
if year==1960:
	bond_target=.0097
	stock_target=0.0733
	mu=0.0194
	delta=0.0158
	sigma=0.15

print("bond_target",bond_target);
print("stock_target",stock_target);
print("pie_target",pie_target);
print("sigma",sigma);
print("mu, delta",mu,delta);


########################################################################
# define model 
########################################################################

PHI= np.zeros((2,2));
I = np.zeros((2,2));
lamda = np.zeros((2));

I[0,0]=1.0;
I[1,1]=1.0;
beta=1/(1+bond_target);
B=np.zeros((2));
W=np.zeros((2));
A=np.zeros((2,2));
re=np.zeros((2,2));
Re=np.zeros((2));
pie=np.zeros((2));
tot_ret=0;
detA=0;



#satisfy serial correlation and target_pi
pie[0]=pie_target;
pie[1]=1-pie[0];


zeta=1+mu;
lamda[1]=zeta - delta/(pie[1]*(1+pie[1]/pie[0]))**0.5;
lamda[0]=zeta-(pie[1]/pie[0])*(lamda[1]-zeta);


print("lamda", lamda);
print("input mean", zeta);
print("calculated mean", np.dot(pie,lamda));
print("input delta",delta)
delta=(np.dot(pie,(lamda-(1+mu))**2))**0.5;
print("calculated delta", delta);

C=np.zeros((4,4));
D=np.zeros((4));

# serial correlation;
for i in range (0,2):
        for j in range(0,2):
                C[0,2*i+j]=pie[i]*(lamda[i]-zeta)*(lamda[j]-zeta);
D[0]=sigma*delta*delta;
# probabilities in each state sum to 1
C[1,0:2]=1; D[1]=1;
C[2,2:4]=1; D[2]=1;
# stationary
C[3,0]=pie[0]; C[3,2]=pie[1];D[3]=pie[0];


E=np.linalg.solve(C,D);

"""
print("C\n",C);
print("D\n",D);
print("detC", np.linalg.det(C));
print("E\n",E);
"""

PHI[0,:]=E[0:2];
PHI[1,:]=E[2:4];


#	run Markov chain ********************
print("seed pie", pie)
if np.linalg.norm(pie-np.matmul(np.transpose(PHI),pie))>0.001:
	print("not stationary ******************",pie)
print("PHI",PHI);

#	verify serial correlation  ************************
t_sigma=0;
for i in range (0,2):
        for j in range(0,2):
                t_sigma+=pie[i]*PHI[i,j]*(lamda[i]-zeta)*(lamda[j]-zeta);
t_sigma=t_sigma/(delta*delta);
print("input sigma", sigma);
print("calculated sigma", t_sigma);
#	verify lamda mean  ************************
lamda_mean=lamda.mean();
print("lamda mean", lamda_mean);



########################################################################
# define model 
########################################################################
		
def calc_w(alpha,print_flag):
	global B,A,W, re,Re,tot_ret, detA;

	A[0,0]=beta*PHI[0,0]*(lamda[0]**(1-alpha));
	A[0,1]=beta*PHI[0,1]*(lamda[1]**(1-alpha));
	A[1,0]=beta*PHI[1,0]*(lamda[0]**(1-alpha));
	A[1,1]=beta*PHI[1,1]*(lamda[1]**(1-alpha));

	B[0]=A[0,0]+A[0,1];
	B[1]=A[1,0]+A[1,1];


	A= I - A;
	detA= np.linalg.det(A);
	if print_flag: print ("detA",detA);
	if detA <=0:
		return(0);

	W=np.linalg.solve(A,B);
	
	re[0,0]=lamda[0]*(W[0]+1)/W[0] - 1;
	re[0,1]=lamda[1]*(W[1]+1)/W[0] - 1;
	re[1,0]=lamda[0]*(W[0]+1)/W[1] - 1;
	re[1,1]=lamda[1]*(W[1]+1)/W[1] - 1;
	if print_flag: print ("re",re);

	Re[0]=np.dot(PHI[0,:],re[0,:]);
	Re[1]=np.dot(PHI[1,:],re[1,:]);
	if print_flag: print ("Re",Re);

	if print_flag: print("W",W);
	tot_ret=np.dot(pie,Re)
	return(1);


########################################################################
# main loop
########################################################################

N=1+(alpha_max-alpha_min)*4;
a0=np.zeros((N));
std=np.ones((N));
ret=np.zeros((N));
rvol=np.zeros((N));

a0_scale=(alpha_max-alpha_min)/(N-1);

for i in range(0,N):
	a0[i]=alpha_min+i*a0_scale;

if verbose>0:
	print ("a0", a0);

if np.min (PHI)<0:
        print("bad PHI");
else:
	for i in range(0,N):
		if calc_w(a0[i],0) >0:
			std_r=0;
			std_r+=pie[0]*PHI[0,0]*(re[0,0]-tot_ret)**2;
			std_r+=pie[0]*PHI[0,1]*(re[0,1]-tot_ret)**2;
			std_r+=pie[1]*PHI[1,0]*(re[1,0]-tot_ret)**2;
			std_r+=pie[1]*PHI[1,1]*(re[1,1]-tot_ret)**2;
			std_r=std_r**0.5;
			std[i]=std_r;
			ret[i]=tot_ret;
			rvol[i]=(tot_ret - bond_target)/std_r;
	
	i=rvol.argmax();
	calc_w(a0[i],1);
	print ("###m",pie_target,a0[i],ret[i],std[i],rvol[i]);
	if verbose>0:	
		for i in range(0,N):
			if ret[i]>0:
				print ("###v",pie_target,a0[i],ret[i],std[i],rvol[i]);

