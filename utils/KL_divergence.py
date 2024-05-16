import torch 

def KL_divergence(P,Q):
        def Gauss_conditional(input):
            batch_size = input.size(0)          
            pji = torch.zeros(batch_size,batch_size)
            #pji.to(self.device)        
            P = input.squeeze()
            print(P)
            cov_P=torch.cov(P.t())
            print(cov_P)
            diff_matrix = (P.unsqueeze(1) - P.unsqueeze(0)).unsqueeze(-1)  # 形状为 [batch_size, batch_size, feature_size]
            print(diff_matrix)
            tem = torch.matmul(diff_matrix.transpose(-1, -2), cov_P)        
            intermediate_matrix = torch.matmul(tem, diff_matrix)  # 形状为 [batch_size, batch_size, feature_size]
            pji = intermediate_matrix.squeeze()
            pji.fill_diagonal_(0)
            print(pji)
            pji = torch.exp(pji * -0.5)  # 形状为 [batch_size, batch_size]
            print(pji)
            pji = pji / torch.sum(pji, dim=1, keepdim=True)  # 形状为 [batch_size, batch_size]
            print(pji)
            
        
            
            return pji
        
        pji = Gauss_conditional(P)

        input("modulepause")
        qji = Gauss_conditional(Q)
        input("modulepause")
        
        kl_di = torch.sum(pji*torch.log(pji/qji))
        print(kl_di)
        
        return kl_di



a = torch.rand(200,1,1600)
b = torch.rand(200,128)
kl = KL_divergence(a,b)