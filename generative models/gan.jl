"""
Generative Adversarial Networks
"""

#importowanie bibliotek
using Pkg; for p in ("Knet","Colors","Images"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Knet, Colors, Images, Statistics

#załadowanie zbioru danych mnist
include(Pkg.dir("Knet","data","mnist.jl")) #MNIST data loader functions
global atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

#zdefiniowanie generatora i dyskryminatora
function mlp(w,x;p=0.0,activation=elu,outputactivation=sigm)
    for i=1:2:length(w)
        x = w[i]*dropout(mat(x),p) .+ w[i+1] # mat() used for flattening images to a vector.
        i<length(w)-1 && (x = activation.(x)) 
    end
    return outputactivation.(x) #output layer
end

"""
global const 𝜀=Float32(1e-8): Definiuje stałą globalną 𝜀 o wartości 1e-8 typu Float32. 𝜀 jest używane w funkcjach straty jako mała wartość zapobiegająca błędowi dzielenia przez zero.
D(w,x;p=0.0) = mlp(w,x;p=p): Definiuje funkcję D, która reprezentuje dyskryminator. Przyjmuje trzy argumenty - wagi w, dane wejściowe x oraz opcjonalny parametr p (prawdopodobieństwo dropoutu). Funkcja ta korzysta z funkcji mlp do utworzenia modelu.
G(w,z;p=0.0) = mlp(w,z;p=p): Definiuje funkcję G, która reprezentuje generator. Podobnie jak funkcja D, przyjmuje trzy argumenty - wagi w, wektor szumów z oraz opcjonalny parametr p. Wykorzystuje funkcję mlp do utworzenia modelu generatora
"""
global const 𝜀=Float32(1e-8)
D(w,x;p=0.0) = mlp(w,x;p=p)
G(w,z;p=0.0) = mlp(w,z;p=p)

#funkcje straty dla dyskryminatora i generatora 
𝑱d(𝗪d,x,Gz) = -mean(log.(D(𝗪d,x) .+ 𝜀)+log.((1+𝜀) .- D(𝗪d,Gz)))/2   
𝑱g(𝗪g, 𝗪d, z) = -mean(log.(D(𝗪d,G(𝗪g,z)) .+ 𝜀))           
𝒩(input, batch) = atype(randn(Float32, input, batch))  #SampleNoise

#funkcja grad z Knet do obliczenia gradientów funkcji straty dla dyskryminatora i generatora.
∇d  = grad(𝑱d) # Discriminator gradient
∇g  = grad(𝑱g) # Generator gradient

#inicjalizacja wag modelu dla warstw ukrytych i warstwy wyjściowej
function initweights(hidden,input, output)
    𝗪 = Any[];
    x = input
    for h in [hidden... output]
        push!(𝗪, atype(xavier(h,x)), atype(zeros(h, 1))) #FC Layers weights and bias
        x = h
    end
    return 𝗪  #return model params
end

#ta funkcja generuje i zapisuje określoną liczbę obrazów przy użyciu modelu generatora
function generate_and_save(𝗪,number,𝞗;fldr="generations/")
    Gz = G(𝗪[1], 𝒩(𝞗[:ginp], number)) .> 0.5
    Gz = permutedims(reshape(Gz,(28,28,number)), (2,1,3))
    [save(fldr*string(i)*".png",Gray.(Gz[:,:,i])) for i=1:number]
end

#trening modelu
function runmodel(𝗪, data, 𝞗; dtst=nothing, optim=nothing, train=false, saveinterval=20)
    gloss = dloss = total=0.0;
    B = 𝞗[:batchsize]
    for i=1:(train ? 𝞗[:epochs] : 1)
        for (x,_) in data
            total+=2B
            Gz = G(𝗪[1], 𝒩(𝞗[:ginp], B)) #Generate Fake Images
            train ? update!(𝗪[2], ∇d(𝗪[2],x,Gz), optim[2]) : (dloss += 2B*𝑱d(𝗪[2], x, Gz))
            
            z=𝒩(𝞗[:ginp],2B) #Sample z from Noise
            train ? update!(𝗪[1], ∇g(𝗪[1], 𝗪[2], z), optim[1]) : (gloss += 2B*𝑱g(𝗪[1],𝗪[2],z))       
        end
        train ? runmodel(𝗪, dtst, 𝞗; train=false) : println((gloss/total, dloss/total))
        i % saveinterval == 0 && generate_and_save(𝗪, 100, 𝞗)  # save 10 images
    end
end

function main()
    𝞗=Dict(:batchsize=>32,:epochs=>80,:ginp=>256,:genh=>[512],:disch=>[512],:optim=>Adam,:lr=>0.0002);
    xtrn,ytrn,xtst,ytst = mnist()
    global dtrn = minibatch(xtrn, ytrn, 𝞗[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, 𝞗[:batchsize]; xtype=atype)
    𝗪  = (𝗪g,𝗪d)  = initweights(𝞗[:genh], 𝞗[:ginp], 784), initweights(𝞗[:disch], 784, 1)
    𝚶  = (𝚶pg,𝚶pd) = optimizers(𝗪g, 𝞗[:optim]; lr=𝞗[:lr]), optimizers(𝗪d,𝞗[:optim]; lr=𝞗[:lr])
    generate_and_show(𝗪,100,𝞗)
    runmodel(𝗪, dtst, 𝞗; optim=𝚶, train=false) # initial losses
    runmodel(𝗪, dtrn, 𝞗; optim=𝚶, train=true, dtst=dtst) # training 
    𝗪,𝚶,𝞗,(dtrn,dtst)    # return weights,optimizers,options and dataset
end

#wywołanie całego skryptu, inicjalizacja modelu, trening i wyniki
main()
