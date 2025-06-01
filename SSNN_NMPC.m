%% SSNN based data-driven NMPC
% Implemented on continuous stirred tank reactor (CSTR) system
% Reference: https://arxiv.org/abs/2505.11524

%% Generating training and testing dataset
clear all;close all
global UD n m p h 
randn('state',2);
rand('state',3);
NT=1000;n=2;m=1; p=1; D=500;  Ds=500;
B = 22.0;Da = 0.082;beta = 3.0; T=0.1;
x=zeros(n,NT+1); x0=[0;0];     x(:,1)=x0;
u=kron(-0.5*rand(1,NT/50),ones(1,50)); 
 for k=1:NT                
    x(:,k+1)=x(:,k)+T*[-x(1,k)+Da*(1-x(1,k))*exp(x(2,k)); -x(2,k) + B*Da*(1-x(1,k))*exp(x(2,k))-beta*(x(2,k)-u(:,k))];  % here f is function corresponding to the state equaton of the system in discrete-time 
    y(:,k)=x(2,k);             
 end                               

UD=u(:,1:500);                               % Training input
YD=y(:,1:500)+0.01*randn(1,500);             % Training output                     
Us=u(:,501:1000);                            % Testing input
Ys=y(:,501:1000)+0.01*randn(1,500);          % Testing output

%% State space model identification using SSNN
h=2;
theta=rand(h,3*h+m+p+2);     % theta contains the hyperparameters in the model
for i=1:2
   % Training the SSNN
   fun = @(theta)trace((YD-theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*f_xp(theta,UD,D)))'*(YD-theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*f_xp(theta,UD,D))));
   options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
   [theta,fval,flag]=fminunc(fun,theta,options);  
end
 
%% Performance evaluation on training and testing data
% Predicting the training output with SSNN    
Xp=zeros(h,D+1);  Xp(:,1)=theta(:,3*h+m+p+2); Yp=zeros(p,D);   
for k=1:D
Xp(:,k+1)=theta(:,h+m+2:2*h+m+1)*tanh(theta(:,1:h)*Xp(:,k)+theta(:,h+1:h+m)*UD(:,k)+theta(:,h+m+1));
Yp(:,k)=theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*Xp(:,k));
end 
%Estimating the initial state
    Xts10=randn(h,1);
    options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e6)
    fun0=@(Xts1)[Ys(:,1)-theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*Xts1); Ys(:,2)-theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*(theta(:,h+m+2:2*h+m+1)*tanh(theta(:,1:h)*Xts1+theta(:,h+1:h+m)*Us(:,1)+theta(:,h+m+1))))];
    [Xts1,fval0,flag0] = fsolve(fun0,Xts10,options1);

% Predicting the testing output with SSNN   
Xsp=zeros(h,Ds);  Xsp(:,1)=Xts1; 
for k=1:Ds
Xsp(:,k+1)=theta(:,h+m+2:2*h+m+1)*tanh(theta(:,1:h)*Xsp(:,k)+theta(:,h+1:h+m)*Us(:,k)+theta(:,h+m+1));
Ysp(:,k)=theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*Xsp(:,k));
end 


%% SSNN-based NMPC
N=10; NS=200; 
for k=1:NS
if k>=1 && k<=50
    Yr(k)=1;
elseif k>50 && k<=100    
    Yr(k)=0.7;
 elseif k>100 && k<=150    
    Yr(k)=0.5;
else    
    Yr(k)=1;
end
      
Xr0=randn(n+m,1);
fun1=@(Xr1)[Yr(k)-theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*(Xr1(1:n,1)));Xr1(1:n,1)-theta(:,h+m+2:2*h+m+1)*tanh(theta(:,1:h)*Xr1(1:n,1)+theta(:,h+1:h+m)*Xr1(n+1:n+m,1)+theta(:,h+m+1))];
[Xr1,fval1,flag1] = fsolve(fun1,Xr0,options1);
Xr(:,k)=Xr1(1:n,1);
Ur(:,k)=Xr1(n+1:n+m,1);
end

Qx=[5 0;0 4];
Qu=2;
X=zeros(n,NS+1); X(:,1)=x0;
Xk=zeros(n,N+1); Xk(:,1)=x0;
U=zeros(m,NS); %u(:,1)=Ur(:,1);
Y=zeros(p,NS);
umin=-2;umax=0.5;
Uk0=-1*rand(m,N);
Uk1=zeros(m,N);
% simulating system with SSNN NMPC 
for k=1:NS
   theta(:,3*h+m+p+2)=X(:,k);   % x_k|k is the current state 
   Xrk=Xr(:,k);Urk=Ur(:,k); Yrk= Yr(:,k);
   fun3 = @(Uk)trace((Xrk-f_xp(theta,Uk,N))'*Qx*(Xrk-f_xp(theta,Uk,N)))+trace((Uk-Uk1)'*Qu*(Uk-Uk1));
   F=[];g=[];Feq=[];geq=[];lb=[umin*ones(m,N)];ub=[umax*ones(m,N)];nonlcon=[];
   Uk=fmincon(fun3,Uk0,F,g,Feq,geq,lb,ub,nonlcon);
   U(k)=Uk(:,1);
   X(:,k+1)=theta(:,h+m+2:2*h+m+1)*tanh(theta(:,1:h)*X(:,k)+theta(:,h+1:h+m)*U(:,k)+theta(:,h+m+1));
   Y(:,k)=theta(:,3*h+m+2:3*h+m+p+1)'*tanh(theta(:,2*h+m+2:3*h+m+1)*X(:,k));
   Uk0=Uk;
   Uk1=Uk;
end   

%% Plotting results
figure(1)
subplot(2,1,1)
plot(UD,'r', LineWidth=3)
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$u_{k}$','Interpreter','latex','fontsize',18);
grid on
subplot(2,1,2)
plot(YD,'r', LineWidth=3)
hold on
plot(Yp,'g', LineWidth=2)
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$y_{k}, \hat{y}_{k}$','Interpreter','latex','fontsize',18);
legend('$y_{k}$', '$\hat{y}_{k}$','Interpreter','latex','fontsize',14)
grid on
figure(2)
plot(Yr,'g', LineWidth=5)
hold on
plot(Y,'r', LineWidth=3)
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$y_{r_k}, y_{k|k}$','Interpreter','latex','fontsize',18);
legend('$y_{r_k}$', '$y_{k|k}$','Interpreter','latex','fontsize',14)
grid on
figure(3)
stairs(U,'r', LineWidth=2)
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$u_{k|k}$','Interpreter','latex','fontsize',18);
grid on


%% Predicting the states using f_xp 
function Xhat=f_xp(theta,u,NP)    
global n m Ts h p
Xhat=zeros(h,NP); Xhat(:,1)=theta(:,3*h+m+p+2);
for i=1:NP-1
    Xhat(:,i+1)=theta(:,h+m+2:2*h+m+1)*tanh(theta(:,1:h)*Xhat(:,i)+theta(:,h+1:h+m)*u(:,i)+theta(:,h+m+1));
end
end
