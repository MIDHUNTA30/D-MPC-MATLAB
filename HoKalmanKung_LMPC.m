%% Ho-Kalman-Kung algorithm based data-driven LMPC
% Reference: https://arxiv.org/abs/2505.11524

%% Generating training and testing dataset
clear all;close all
% Generate the input-output data by simulating the linear system
Ad = [0.2 -0.4 0.5;0.7 0.3 0.6;0.-0.5 0.1 0.6];Bd=[0.1;0.2;0.1];Cd = [1 0 0];
n=width(Ad);m=width(Bd);p=height(Cd);
D=50;
UD =zeros(m,D); UD(:,1)=1;  % define the pulse input
XD =zeros(n,D+1); YD=zeros(p,D);
for k=1:D
XD(:,k+1)=Ad*XD(:,k)+Bd*UD(:,k);
YD(:,k) = Cd*XD(:,k);   
end

% Constructing Hankel matrix using output data
N=5;H=10; 
Hy1= f_Hankel(YD,N,H,p);               
[W,S,V] =svd(Hy1);
epsilon=0.005;
for i=1:N
    if S(i,i) >= epsilon
        n=i;
    end
end    
Wn= W(:,1:n);Sn =S(1:n,1:n);Vn =V(:,1:n); Pn=eye(n);
On= Wn*sqrtm(Sn)*Pn;Cn =pinv(Pn)*sqrtm(Sn)*Vn';
A=pinv(On(1:N-1,:))*On(2:N,:); B = Cn(:,1); C = On(1,:);  % Compute A,B,C matrices using On and Cn     % A=pinv(On(1:N-1,:))*On(2:N,:);

% Predicting output with identified model
Xhat =zeros(n,D+1); Yhat=zeros(p,D);
for k=1:D
Xhat(:,k+1)=A*Xhat(:,k)+B*UD(:,k);
Yhat(:,k) = C*Xhat(:,k);   
end

% Parameters for D-LMPC
NT=50;N=10;
xmin=-10;xmax=10;umin=-1;umax=1;

x0=[10;5;2]; 
X=zeros(n,NT+1); X(:,1)=x0;
Xk=zeros(n*(N+1),1); Xk(1:n,1)=x0;
U=zeros(m,NT);
Uk=zeros(m*N,1);
zk=[Xk;Uk];

% constructing AX,BU,QX,RU,H
for i=1:N+1
    AX((i-1)*n+1:i*n,:)=A^(i-1);
end
for i=1:N+1
  for j=1:N
      if i>j
          BU((i-1)*n+1:i*n,(j-1)*m+1:j*m)=A^(i-j-1)*B;
      else
          BU((i-1)*n+1:i*n,(j-1)*m+1:j*m)=zeros(n,m);
      end    
  end
end
Qx=eye(n); QxN=Qx; Ru=0.5*eye(m);
QX=Qx;RU=Ru;
for i=1:N-1
  QX=blkdiag(QX,Qx); RU=blkdiag(RU,Ru);
end
QX=blkdiag(QX,QxN);
H=blkdiag(QX,RU);

% Simulating system with MPC
for k=1:NT
   xk=X(:,k);  
   fun = @(z)z'*H*z;
   F=[];g=[];Feq=[eye((N+1)*n) -BU];geq=AX*xk;
   lb=[xmin*ones(1,(N+1)*n),umin*ones(1,N*m)];
   ub=[xmax*ones(1,(N+1)*n),umax*ones(1,N*m)];
   z=fmincon(fun,zk,F,g,Feq,geq,lb,ub);
   U(:,k)=z((N+1)*n+1:(N+1)*n+m,1);
   X(:,k+1)=A*X(:,k)+B*U(:,k);
   zk=z;
end    

% plotting response
figure(1)
subplot(2,1,1)
time = (0:NT);
stairs(time(1:end-1),UD,'r.-','LineWidth',2) 
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$u_{k}$','Interpreter','latex','FontSize',18);
grid on
subplot(2,1,2)
plot(time(1:end-1),YD,'r.-','LineWidth',2) 
hold on
plot(time(1:end-1),Yhat,'g.-','LineWidth',2) 
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$y_{k}, \hat{y}_{k}$','Interpreter','latex','fontsize',18);
legend('$y_{k}$', '$\hat{y}_{k}$','Interpreter','latex','fontsize',14)
grid on
figure(2)
time = (0:NT);
plot(time,X(1,:),'r.-','LineWidth',2) 
hold on
plot(time,X(2,:),'g.-','LineWidth',2) 
hold on
plot(time,X(3,:),'k.-','LineWidth',2) 
legend('$x_1$','$x_2$','$x_3$','Interpreter','latex','fontsize',14);
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$\textbf{x}_{k|k}$','Interpreter','latex','FontSize',18);
grid on
figure(3)
stairs(time(1:end-1),U,'r.-','LineWidth',2)
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('${u}_{k|k}$','Interpreter','latex','FontSize',18);
grid on

% Function to compute block Hankel matrix 
function Hv=f_Hankel(v,N,H,q)    
Hv=zeros(N*q,H);
for i=1:N
    for j=1:H
    Hv((i-1)*q+1:i*q,j)=v(:,i+j-1);
    end
end
end