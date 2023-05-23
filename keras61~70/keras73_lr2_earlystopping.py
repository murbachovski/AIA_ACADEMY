x = 10
y = 10
w = 42
lr = 0.01
epochs = 50000
patience = 100  # Number of epochs to wait for improvement
best_loss = float('inf')
no_improvement_counter = 0

for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2  # MSE

    print(i, 'LOSS : ', round(loss, 4), '\nPredict : ', round(hypothesis, 4))

    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2

    down_predict =  x * (w - lr)
    down_loss = (y - down_predict) ** 2

    if up_loss >= down_loss:
        w = w - lr
    else: 
        w = w + lr

    # Check for early stopping
    if loss < best_loss:
        best_loss = loss
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1

    if no_improvement_counter >= patience:
        print('Early stopping at epoch', i)
        break
