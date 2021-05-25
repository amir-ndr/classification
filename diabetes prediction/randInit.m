function init = randInit(input,out)
  
init = zeros(out,1+input);

epsilon = sqrt(6)/(sqrt(input+out));
init = rand(out,1+input) * 2 * epsilon -epsilon;
end
