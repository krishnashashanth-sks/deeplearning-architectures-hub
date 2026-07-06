import torch
import torch.optim as optim
import torch.nn as nn

def adapt_and_evaluate(test_loader_for_ttt,CustomCNN,adapt_steps,adapt_lr,ss_criterion,device):
    all_preds = []
    all_targets = []

    print("Starting test-time adaptation loop...")

    # 5. Iterate through each batch in test_loader_for_ttt
    for batch_idx, (original_img, target, ss_img, ss_label) in enumerate(test_loader_for_ttt):
        # a. Move data to device
        original_img, target = original_img.to(device), target.to(device)
        ss_img, ss_label = ss_img.to(device), ss_label.to(device)

        # b. Adapt the model for the current batch: Load original supervised_model.pth for each new batch
        model_for_adaptation = CustomCNN(num_classes=10, num_ss_classes=4) # Instantiate a fresh model
        model_for_adaptation.load_state_dict(torch.load('supervised_model.pth')) # Load supervised weights
        model_for_adaptation.to(device)

        # c. Set model to eval() mode but BatchNorm2d layers to train() mode
        model_for_adaptation.eval() # Disable dropout, etc.
        for m in model_for_adaptation.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train() # Enable BatchNorm to update running statistics

        # d. Freeze gradients for all parameters, then enable for self_supervised_head and BatchNorm affine parameters
        for param in model_for_adaptation.parameters():
            param.requires_grad = False

        for param in model_for_adaptation.self_supervised_head.parameters():
            param.requires_grad = True

        for m in model_for_adaptation.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None: # Ensure weight exists (e.g., for LayerNorm no weight)
                    m.weight.requires_grad = True
                if m.bias is not None: # Ensure bias exists
                    m.bias.requires_grad = True

        # e. Create a new optimizer for adaptation
        adapter_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model_for_adaptation.parameters()),
            lr=adapt_lr
        )

        # f. Perform adapt_steps optimization steps
        for step in range(adapt_steps):
            adapter_optimizer.zero_grad()

            # Forward pass for self-supervised task
            # We only need the ss_output for adaptation
            _, ss_output = model_for_adaptation(ss_img)
            ss_loss = ss_criterion(ss_output, ss_label)

            ss_loss.backward()
            adapter_optimizer.step()

        # g. Evaluate on the adapted model
        model_for_adaptation.eval() # Set model back to eval to use adapted BN statistics for classification
        with torch.no_grad(): # No need to calculate gradients for final classification prediction
            cls_output, _ = model_for_adaptation(original_img)
            preds = torch.argmax(cls_output, dim=1)

        # h. Store the predicted labels and true labels
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader_for_ttt)} batches for TTT.")
    print("Finished test-time adaptation and evaluation.")
    return all_preds,all_targets


def adapt_and_evaluate_with_entropy(test_loader_for_ttt,CustomCNN,adapt_lr,adapt_steps,entropy_loss_fn,ss_criterion,entropy_weight,device):
    all_preds_entropy_ttt = []
    all_targets_entropy_ttt = []

    print("Starting test-time adaptation loop with Entropy Minimization...")

    #  Iterate through each batch in test_loader_for_ttt
    for batch_idx, (original_img, target, ss_img, ss_label) in enumerate(test_loader_for_ttt):
        # a. Move data to device
        original_img, target = original_img.to(device), target.to(device)
        ss_img, ss_label = ss_img.to(device), ss_label.to(device)

        # b. Adapt the model for the current batch: Load original supervised_model.pth for each new batch
        # Ensure CustomCNN class is defined in the environment (from previous step)
        model_for_adaptation = CustomCNN(num_classes=10, num_ss_classes=4) # Instantiate a fresh model
        model_for_adaptation.load_state_dict(torch.load('supervised_model.pth')) # Load supervised weights
        model_for_adaptation.to(device)

        # c. Set model to eval() mode but BatchNorm2d layers to train() mode
        model_for_adaptation.eval() # Disable dropout, etc.
        for m in model_for_adaptation.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train() # Enable BatchNorm to update running statistics

        # d. Freeze gradients for all parameters, then enable for self_supervised_head, classifier, and BatchNorm affine parameters
        for param in model_for_adaptation.parameters():
            param.requires_grad = False

        for param in model_for_adaptation.self_supervised_head.parameters():
            param.requires_grad = True

        for param in model_for_adaptation.classifier.parameters(): # Enable for classifier head
            param.requires_grad = True

        for m in model_for_adaptation.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True

        # e. Create a new optimizer for adaptation
        adapter_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model_for_adaptation.parameters()),
            lr=adapt_lr
        )

        # f. Perform adapt_steps optimization steps
        for step in range(adapt_steps):
            adapter_optimizer.zero_grad()

            # Forward pass for self-supervised task and classification (for entropy)
            cls_output, ss_output = model_for_adaptation(original_img) # Use original_img for classification path

            # Calculate self-supervised loss
            ss_loss = ss_criterion(ss_output, ss_label)

            # Calculate entropy loss
            entropy_loss = entropy_loss_fn(cls_output)

            # Combine losses
            combined_loss = ss_loss + entropy_loss * entropy_weight

            combined_loss.backward()
            adapter_optimizer.step()

        # g. Evaluate on the adapted model
        model_for_adaptation.eval() # Set model back to eval to use adapted BN statistics for classification
        with torch.no_grad(): # No need to calculate gradients for final classification prediction
            cls_output, _ = model_for_adaptation(original_img)
            preds = torch.argmax(cls_output, dim=1)

        # h. Store the predicted labels and true labels
        all_preds_entropy_ttt.extend(preds.cpu().numpy())
        all_targets_entropy_ttt.extend(target.cpu().numpy())

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader_for_ttt)} batches for TTT with entropy minimization.")
    print("Finished test-time adaptation and evaluation with Entropy Minimization.")
    return all_preds_entropy_ttt,all_targets_entropy_ttt    
