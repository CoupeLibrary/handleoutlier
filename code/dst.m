function b = dst(a,n)
% DST    Discrete sine tranform         (Used in Poisson reconstruction)
%    Y = DST(X) returns the discrete sine transform of X.
%    The vector Y is the same size as X and contains the
%    discrete sine transform coefficients.
%    Y = DST(X,N) pads or truncates the vector X to length N
%    before transforming.
%    If X is a matrix, the DST operation is applied to each
%    column. This transform can be inverted using IDST.

error(nargchk(1,2,nargin));

if min(size(a)) == 1
    if size(a,2)>1
        do_trans = 1;
    else
        do_trans = 0;
    end
    a = a(:);
else
    do_trans = 0;
end
if nargin == 1,     n = size(a,1);      end
m = size(a,2);

% Pad or truncate a if necessary
if size(a,1)<n,
    aa = zeros(n,m);        aa(1:size(a,1),:) = a;
else
    aa = a(1:n,:);
end

y = zeros(2*(n+1),m); y(2:n+1,:)=aa;    y(n+3:2*(n+1),:)=-flipud(aa);
yy = fft(y);        b=yy(2:n+1,:)/(-2*sqrt(-1));

if isreal(a), b = real(b); end;
if do_trans, b = b.'; end
