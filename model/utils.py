def count_parameters(model):
    parametrs_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parametrs_num} ({parametrs_num / (10 ** 6):.1f}M)")