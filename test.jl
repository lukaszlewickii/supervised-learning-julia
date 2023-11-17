using Flux
using Flux: gradient, update!

#training data
X = rand(10, 2)
y = 2 * X[:, 1] + 3 * X[:, 2] + 0.1 * randn(10)

model = Chain(Dense(2, 1))

loss(x, y) = Flux.mse(model(x), y)

η = 0.01
epochs = 100

opt = Descent(η)

dataset = [(X[i, :], y[i]) for i in 1:size(X, 1)]
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), dataset, opt)
    println("Epoch $epoch, Loss: $(loss(X, y))")
end

println("Wagi modelu po uczeniu: $(Flux.params(model))")