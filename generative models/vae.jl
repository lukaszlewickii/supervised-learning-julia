
"""
Implementacja Variational Auto-Encoder w języku Julia
MIT License
"""

#importowanie bibliotek
using Flux, Random
using Flux: logitbinarycrossentropy, chunk
using Flux.Optimise
using Flux.Data
using Flux: throttle
using Statistics
using Images

#definicja wymiarów wejściowych (x_dim, h_dim), wymiarów przstrzeni ukrytej (z_dim) oraz tablica warstw (layers)
x_dim = 784
h_dim = 256
z_dim = 5
layers = Array{Dense}(undef, 5)

#parametry uczenia
batch_size = 100
sample_size = 10
data_dir = "C:\\Users\\Łukasz\\III semestr infa\\supervised-learning-julia\\generative models\\data"
project_dir = "C:\\Users\\Łukasz\\III semestr infa\\supervised-learning-julia\\generative models\\data\\outputs"
train_data_filename = "train-images.idx3-ubyte"

#załadowanie zbioru danych mnist
function load_images(dir)
    filepath = joinpath(dir, train_data_filename)
    io = IOBuffer(read(filepath))
    _, N, nrows, ncols = MNIST.imageheader(io)
    images = [MNIST.rawimage(io) for _ in 1:N]
    # Transform into (784, N) in shape and change each data element into Float32
    return Float32.(hcat(vec.(images)...))
end
images = load_images(data_dir)
nobs = size(images, 2)

#podział całego zbioru danych na mniejsze batche
data = [images[:, i] for i in Iterators.partition(1:nobs, batch_size)]

#zapisywanie obrazów
function save_image(batch_data, filename)
    chunked_data = chunk(batch_data, sample_size)
    im_data = reshape.(chunked_data, 28, :)
    im = Gray.(vcat(im_data...))
    image_path = joinpath(project_dir, filename)
    save(image_path, im)
end

#przykład mini-batcha jako reconstruction target 
sample_data = data[1]
save_image(sample_data, "mnist_base.png")

#inicjalizacja sieci neuronowej i poszczególnych warstw
layers[1] = Dense(x_dim, h_dim, relu)
layers[2] = Dense(h_dim, z_dim)
layers[3] = Dense(h_dim, z_dim)

#implementacja kodera, przyjmuje dane wejściowe -> przekształca je przez pierwszą warstwę sieci -> zwraca 2 gałęzie wyjściowe - outputy z drugiej i trzeciej warstwy
function g(x)
    h = layers[1](x)
    return (layers[2](h), layers[3](h))
end

#funkcja przyjmująca parametry rozkładu normalnego (mu - wartość oczekiwana, logsig - logarytm wariancji), niezbędne w algorytmie vae
function z(mu, logsig)
    sigma = exp.(logsig / 2)
    return mu + randn(Float32) .* sigma
end

#dekoder - warstwy 4 i 5 tworzą sieć dekodującą, która odbiera wektor z przestrzeni ukrytej i stara się zrekonstruować dane wejściowe
layers[4] = Dense(z_dim, h_dim, relu)
layers[5] = Dense(h_dim, x_dim)
decode(z) = Chain(layers[4], layers[5])(z)

#KL loss - funkcja obliczająca stratę Kullbacka-Leibera dla rozkładu normalnego, element funkcji straty vae, który zapewnia, że przestrzeń ukryta jest zbliżona do standardowej przestrzeni normalnej
loss_kl(mu, logsig) = 0.5 * sum(mu .^ 2 + exp.(logsig) - logsig .- 1, dims=1)

#Reconstruction loss - funkcja obliczająca stratę rekonstukcji, mierząc jak dobrze zdekodowana próbka odpowiada danym wejściowym
loss_reconstruct(x, z) = sum(logitbinarycrossentropy.(decode(z), x), dims=1)

#całkowita funkcja straty sumująca KL i reconstruction loss
function loss(x)
    miu, logsig = g(x)
    encoded_z = z(miu, logsig)
    kl = loss_kl(miu, logsig)
    rec = loss_reconstruct(x, encoded_z)
    mean(kl + rec)
end

#funkcja reconstruct przyjmuje dane wejściowe x, przepuszcza je przez kodera, a następnie dekoduje uzyskaną przestrzeń ukrytą. Rezultatem jest zrekonstruowany obraz.
function reconstruct(x)
    miu, logsig = g(x)
    encoded_z = z(miu, logsig)
    decoded_x = decode(encoded_z)
    sigmoid.(decoded_x)
end

#tutaj tworzony jest optymalizator Adam oraz zbierane są parametry sieci (layers[1:5])
opt = ADAM()
ps = Flux.params(layers[1:5]...)

# obliczanie funkcji straty na losowej mini-batch danych co 30 sekund w trakcie trenowania modelu. Pozwala to na monitorowanie postępu w trakcie uczenia, pokazując, jak funkcja straty ewoluuje w czasie
evalcb = throttle(() -> @show(loss(images[:, rand(1:nobs, batch_size)])), 30)

#pętla trenuje model przez 50 epok. Po każdej epoce rekonstruuje próbkę i zapisuje zrekonstruowany obraz
for epoch in 1:50
    @info "Epoch $epoch"
    train!(loss, ps, zip(data), opt, cb=evalcb)

    #zapisanie przykładowego zrekonstruowanego obrazka po każdej epoce treningu
    decoded_sample = reconstruct(sample_data)
    save_image(decoded_sample, "decode_sample_$epoch.png")
end