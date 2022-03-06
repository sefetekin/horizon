import pdb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
## val and autotest datas are added to training. val-3 is added to testing 
## ALL F1 >= 0.75    
trainset = torchvision.datasets.ImageFolder(
    root='knee 01-34/train', 
    transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(degrees=0,shear=0.25),
        torchvision.transforms.ToTensor()
    ])  
) #this loads all the images and turns them into numbers so that computer can understand it
testset = torchvision.datasets.ImageFolder(
    root='knee 01-34/test', 
    transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine(degrees=0,shear=0.25),
        torchvision.transforms.ToTensor()
    ])  
)
#5.8 epochs will take 2 hours and 10 minutes
writer = SummaryWriter("sheet2/HingeEmbeddingLoss")

weights = []
for path, label in trainset.imgs:
    if label == 0:
        weights.append(3332/3332)
    elif label == 1:
        weights.append(3332/1516)
    elif label == 2:
        weights.append(3332/930)
    
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(weights))
#train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=15, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=15, sampler=sampler)

weights_test = []
for path, label in testset.imgs:
    if label == 0:
        weights_test.append(935/935)
    elif label == 1:
        weights_test.append(935/447)
    elif label == 2:
        weights_test.append(935/274)
    
sampler_test = torch.utils.data.sampler.WeightedRandomSampler(weights_test, num_samples=len(weights_test))
#train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=15, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=15, sampler=sampler_test)
#this shuffles the set and then gets the images(demonstrated as numbers ) in batches of 8 
#test_dataloader = torch.utils.data.DataLoader(testset, batch_size=15, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn_neural_network =  torchvision.models.googlenet(pretrained=True)
cnn_neural_network.fc = torch.nn.Linear(in_features=1024, out_features=3, bias=True)
cnn_neural_network.to(device)
#infuture is 
#outfeature is benign or mallignant
cost_function = nn.HingeEmbeddingLoss().to(device)
#creates a cost funtion(least squares )
gradient_descent = optim.Adam(cnn_neural_network.parameters(),weight_decay=0.001) 
#starts a gradiant descent with a learning range of 0.001
#F1 FOR BOTH TRAINING AND TESTING SHOULD BE OVER 0.7, TRY NEW WAYS
count = 0
for epoch in range(18):  # loop over the dataset multiple times to make it more precia=
    for training_data in train_dataloader: #traverse over the batches
        # get the inputs; data is a list of [inputs(images), labels(benign or mallignant)]
        count+=1
        #count the number of data     

        training_data_inputs, training_data_labels = training_data
        training_data_inputs, training_data_labels = training_data_inputs.to(device), training_data_labels.to(device)
        # zero the parameter gradients
        gradient_descent.zero_grad()
        
        # forward + backward + optimize
        model_predictions = cnn_neural_network(training_data_inputs)
        
        #recreate the cost function according to the new data loaded
        precision, recall, f1, support = precision_recall_fscore_support(training_data_labels.to('cpu'),
                                                torch.argmax(model_predictions.to('cpu'), dim=1),
                                                zero_division=0,
                                                labels=(0,1,2))
        cost = cost_function(model_predictions, training_data_labels)
        writer.add_scalar('training loss', cost, count)
        for index, label in enumerate(f1):
                writer.add_scalar('training_f1_label_'+str(index), label, count)
        cost.backward()
        
        gradient_descent.step()

        #starts testing in every fifth data
        if count % 5 == 0:
            #testing mode
            cnn_neural_network.eval()

            #get the images and labels
            test_data_inputs, test_data_labels = iter(test_dataloader).next()
            test_data_inputs, test_data_labels = test_data_inputs.to(device), test_data_labels.to(device)
            #create the model according to test data
            model_predictions_test = cnn_neural_network(test_data_inputs)
            #cost function of the test data
            
            cost_test = cost_function(model_predictions_test, test_data_labels)
            #back to training mode
            precision, recall, test_f1, support = precision_recall_fscore_support(test_data_labels.to('cpu'),
                                                torch.argmax(model_predictions_test.to('cpu'), dim=1),
                                                zero_division=0,
                                                labels=(0,1,2))
            cnn_neural_network.train()
            writer.add_scalar('testing_cost', cost_test, count)
            for index, label in enumerate(test_f1):
                writer.add_scalar('testing_f1_label_'+str(index), label, count)
                writer.add_scalar('train-test_f1_label_'+str(index), f1[index]-label, count)


writer.close()

print('Finished Training')


