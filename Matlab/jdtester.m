clc
clf

% Builds test matrix
N=500;
Bneg=-1+2*rand(N,N);
A=(Bneg+transpose(Bneg))/2;

% Constructs initial vector
which_state=80;
[E,V]=eig(A);
eigs=diag(V);
sought_state=E(:,which_state);
sought_lambda=eigs(which_state);
error=0.2*ones(N,1);
guess=(sought_state+error)/norm(error+sought_state);
guess_accuracy=abs(transpose(sought_state)*guess);

init_lambda=transpose(guess)*A*guess;










[lambda, e, res, M] = JD_safe(A,guess);
[lambda_guided, e_guided, res, prev_theta, count, V] = JD(A,guess);


subplot(3,1,1)
plot(count+1,lambda_guided,'o')
hold on
plot(count+1,init_lambda,'og')
hold on
plot(count+1,sought_lambda,'*r')
plot(prev_theta,'k')
plot(count+1,eigs,'.k')
hold on

subplot(3,1,2)
plot(e_guided,'r')
hold on
%plot(largest_overlapp,'*g')
plot(E(:,which_state))

subplot(3,1,3)
plot(log10(res))
%plot(largest_overlapp+e_guided,'k')
% for i=1:N
%     subplot(2,1,2)
%     plot(E(:,i))
%     hold on
% end

% subplot(4,1,4)
% plot(V'*e_guided,'r')
% hold on
% plot(V'*sought_state)
%diff=lambda-lambda_guided


shg