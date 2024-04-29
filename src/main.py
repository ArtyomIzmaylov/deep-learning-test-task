from src.config import DEVICE

train_files, val_files = train_test_split(train_val_files, test_size=0.20, stratify=train_val_labels)
val_dataset = SimpsonsDataset(val_files, mode='val')
fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8), \
                        sharey=True, sharex=True)

for fig_x in ax.flatten():
    random_characters = int(np.random.uniform(0,1000))
    im_val, label = val_dataset[random_characters]
    img_label = " ".join(map(lambda x: x.capitalize(),\
                val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
    imshow(im_val.data.cpu(), \
          title=img_label,plt_ax=fig_x)


N_CLASSES = len(np.unique(train_val_labels))

if val_dataset is None:
    val_dataset = SimpsonsDataset(val_files, mode='val')

train_dataset = SimpsonsDataset(train_files, mode='train')

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, N_CLASSES)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters())

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
for param in model.parameters():
    param.requires_grad = True

history_fine_tune = train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, criterion=criterion,
                          epochs=40, batch_size=128, optimizer=optimizer, scheduler=scheduler)


loss, acc, val_loss, val_acc = zip(*history_fine_tune)

plt.figure(figsize=(15, 9))
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.legend(loc='best')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

my_submit.to_csv('my_submission.csv', index=False)
