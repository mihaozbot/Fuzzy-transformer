import torch

def check_gradient_health(model, large_threshold=1e6, small_threshold=1e-6):
    unhealthy_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            max_val = torch.max(torch.abs(param.grad))
            avg_val = torch.mean(torch.abs(param.grad))
            finite_vals = torch.isfinite(param.grad).all()
            
            if not finite_vals:
                unhealthy_gradients = True
                print(f"")
                print(f"Warning: Non-finite gradient values detected in {name}.")
                print(f"Gradient health for {name}: Max abs val {max_val}, Avg abs val {avg_val}, All finite values {finite_vals}")
            if max_val > large_threshold:
                unhealthy_gradients = True
                print(f"Warning: Large gradient values detected in {name}.")
                print(f"Gradient health for {name}: Max abs val {max_val}, Avg abs val {avg_val}, All finite values {finite_vals}")
            if avg_val < small_threshold:
                unhealthy_gradients = True
                print(f"Warning: Small gradient values detected in {name}.")
                print(f"Gradient health for {name}: Max abs val {max_val}, Avg abs val {avg_val}, All finite values {finite_vals}")
                
            
    
    if unhealthy_gradients:
        print("Warning: Unhealthy gradients detected.")
    else:
        print("All gradients are healthy.")
