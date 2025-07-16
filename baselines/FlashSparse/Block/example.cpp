#include <torch/torch.h>
#include <omp.h>
#include <chrono>

struct Value_csr {
    //8x16
    std::vector<int> value_csr;
    //8x8
    std::vector<int> value_templete_csr;
    //8x4
    std::vector<int> value_templete_csr2;
    std::vector<int> colum;
    int row;
    int pad;
};

std::vector<torch::Tensor> blockProcess8_16_csr(torch::Tensor row1, torch::Tensor column1)
{
    // std::cout<<"---DataBlock Process--- "<< std::endl;
    // std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // test[0]=6;
    int rows=row1.size(0)-1;
    int rowsNew=rows/8;
    // std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
    //最终的map
    std::map<int, Value_csr> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_csr v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*8]], &column[row[i*8+8]], std::inserter(mergedSet, mergedSet.end()));
        //填充value值
        v.row=mergedSet.size();
        v.pad=((v.row/16+1))*16;
        if(v.row%16==0)  v.pad-=16;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        // std::sort(mergedVector.begin(), mergedVector.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        // int r=v.row;
        // while(v.pad-r>0)
        // {
        // v.colum.push_back(-1);
        // r++;
        // }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<int> demo_csr(v.row*8, int(-1));
        std::vector<int> demo_templete_csr(v.row*8, int(-1));
        std::vector<int> demo_templete_csr2(v.row*8, int(-1));
        // std::vector<at::Half> demo(column1.size(0));
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = v.pad / 8;
        int p=0;
        //遍历每一行(共8行)
        for(int j=i*8;j<(i+1)*8;j++)
        {
            //遍历每行的每个元素
            for(int m=row[j];m<row[j+1];m++)
            {
                //8x16
                int bId=colmap[column[m]]/16;
                int bInId=colmap[column[m]]%16;
                //写入对应csr的索引
                if(v.row%16==0){
                    demo_csr[bId*128 + p*16 +bInId]=m;
                }else{
                    if(bId < (bIds-1))
                        demo_csr[bId*128 + p*16 +bInId]=m;
                    else
                        demo_csr[bId*128 + p*(v.row%16) +bInId]=m;
                }
                //8x8
                int bId1=colmap[column[m]]/8;
                int bInId1=colmap[column[m]]%8;
                if(v.row%8==0){
                    demo_csr[bId*64 + p*8 +bInId]=m;
                }else{
                    if(bId < (bIds-1))
                        demo_csr[bId*64 + p*8 +bInId]=m;
                    else
                        demo_csr[bId*64 + p*(v.row%8) +bInId]=m;
                }
                // if(bId < (bIds-1))
                //     demo_templete_csr[bId1*64 + p*8 +bInId1]=m;
                // else
                //     demo_templete_csr[bId1*64 + p*(v.row%8) +bInId1]=m;
                //8x4
                int bId2=colmap[column[m]]/4;
                int bInId2=colmap[column[m]]%4;
                if(v.row%4==0){
                    demo_csr[bId*32 + p*4 +bInId]=m;
                }else{
                    if(bId < (bIds-1))
                        demo_csr[bId*32 + p*4 +bInId]=m;
                    else
                        demo_csr[bId*32 + p*(v.row%4) +bInId]=m;
                }
                // if(bId < (bIds-1))     
                //     demo_templete_csr2[bId2*32 + p*4 +bInId2]=m;
                // else
                //     demo_templete_csr2[bId2*32 + p*(v.row%4) +bInId2]=m;

            }
            p++;
        }
        v.value_csr=demo_csr;
        v.value_templete_csr=demo_templete_csr;
        v.value_templete_csr2=demo_templete_csr2;
        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<int> valueNew;
    std::vector<int> valueNew_templete;
    std::vector<int> valueNew_templete2;
    //按顺序整合res
    for (const auto& pair : res) {
     rowNew.push_back(rowNew.back()+pair.second.row);
    //  rowNew.push_back(rowNew.back()+pair.second.pad-pair.second.row);
     colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
     valueNew.insert(valueNew.end(),pair.second.value_csr.begin(),pair.second.value_csr.end());
     valueNew_templete.insert(valueNew_templete.end(),pair.second.value_templete_csr.begin(),pair.second.value_templete_csr.end());
     valueNew_templete2.insert(valueNew_templete2.end(),pair.second.value_templete_csr2.begin(),pair.second.value_templete_csr2.end());
    }
    // for(auto i:rowNew)
    // std::cout<<i<<" ";
    // std::cout<<std::endl;
    //  for(auto i:colNew)
    // std::cout<<i<<" ";
    // std::cout<<std::endl;

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kInt32);
    auto valueTensor2 = torch::from_blob(valueNew_templete.data(), valueNew_templete.size(), torch::kInt32);
    auto valueTensor3 = torch::from_blob(valueNew_templete2.data(), valueNew_templete2.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    
    torch::Tensor valueTensor_templete = torch::empty_like(valueTensor2);
    valueTensor_templete.copy_(valueTensor2);
    torch::Tensor valueTensor_templete2 = torch::empty_like(valueTensor3);
    valueTensor_templete2.copy_(valueTensor3);
    // valueTensor : 8x16
    // valueTensor_templete : 8x8
    // valueTensor_templete2 : 8x4
    return {rowTensor,colTensor,valueTensor,valueTensor_templete,valueTensor_templete2};
}


//new
struct Value_fp16 {
    std::vector<at::Half> value;
    std::vector<int> colum;
    int row;
    int pad;
};

struct Value_tf32 {
    std::vector<float> value;
    std::vector<int> colum;
    int row;
    int pad;
};

struct Value_templete {
    std::vector<int> value_templete;
    std::vector<int> colum;
    int row;
    int pad;
};


std::vector<torch::Tensor> blockProcess_fp16(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1)
{
    // auto start = std::chrono::high_resolution_clock::now();
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<at::Half, 1>();

    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_fp16> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_fp16 v;

        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]); 
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        //获取countVector的长度
        v.row = countVector.size();
        v.pad=((v.row/wide+1))*wide;
        if(v.row%wide==0)  v.pad-=wide;
        
        if(v.row>0){
            std::map<int, int> colmap;
            int c=0;
            for(auto col:countVector){
                v.colum.push_back(col.first);
                colmap[col.first]=c++;
            }

            std::vector<at::Half> demo;
            demo.resize(v.row * window);
            int bIds = v.pad / wide;
            int p=0;
            for(int j=i*window;j<(i+1)*window;j++)
            {
                for(int m=row[j];m<row[j+1];m++)
                {
                    //存储按8列为一个block存储
                    int bId=colmap[column[m]]/wide;
                    int bInId=colmap[column[m]]%wide;
                    if(v.row%wide==0){
                        demo[bId*window*wide + p*wide +bInId]=degree[m];
                    }else{
                        if(bId < (bIds-1))
                            demo[bId*window*wide + p*wide +bInId]=degree[m];
                        else
                            demo[bId*window*wide + p*(v.row%wide) +bInId]=degree[m];
                    }
                }
                p++;
            }
            v.value=demo;
        }
        //封装v
        #pragma omp critical  
        {  res[i]=v;}
        
    }

    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<at::Half> valueNew;
    //按顺序整合res
    for (const auto& pair : res) {
     rowNew.push_back(rowNew.back()+pair.second.row);
     colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
     valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
    }


    // // 获取结束时间点
    // auto end = std::chrono::high_resolution_clock::now();

    // // 计算耗时
    // std::chrono::duration<double> elapsed = end - start;

    // std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat16);
    
    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    
    return {rowTensor,colTensor,valueTensor};
}

std::vector<torch::Tensor> blockProcess_fp16_ori(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1)
{
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<at::Half, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_fp16> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_fp16 v;

        //对column进行合并
        std::vector<int> mergedVector(&column[row[i*window]], &column[row[i*window+window]]); 
        // 使用 unordered_map 统计 vector 中每个元素出现次数
        std::unordered_map<int, int> elementCounts;
        for (const auto& element : mergedVector) {
            elementCounts[element]++;
        }
        // 将统计结果放入 vector<pair> 中
        std::vector<std::pair<int, int>> countVector(elementCounts.begin(), elementCounts.end());
        std::sort(countVector.begin(), countVector.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        //获取countVector的长度
        v.row = countVector.size();
        v.pad=((v.row/wide+1))*wide;
        if(v.row%wide==0)  v.pad-=wide;

        std::map<int, int> colmap;
        int c=0;
        for(auto col:countVector){
            v.colum.push_back(col.first);
            colmap[col.first]=c++;
        }

        if(v.pad>0){
            for(int j=0; j<v.pad; j++)
                v.colum.push_back(-1);
        }

        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<at::Half> demo(v.pad*window);
        
        //在有值的位置写入相应的value值
        int bIds = v.pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                demo[bId*window*wide + p*wide +bInId]=degree[m];

            }
            p++;
        }
        v.value=demo;
        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<at::Half> valueNew;
    //按顺序整合res
    for (const auto& pair : res) {
     rowNew.push_back(rowNew.back()+pair.second.row);
     rowNew.push_back(rowNew.back()+pair.second.pad-pair.second.row);
     colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
     valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
    }

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat16);
    
    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    
    return {rowTensor,colTensor,valueTensor};
}


std::vector<torch::Tensor> blockProcess_tf32(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1)
{
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_tf32> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_tf32 v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        v.row=mergedSet.size();
        v.pad=((v.row/wide+1))*wide;
        if(v.row%wide==0)  v.pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        // int r=v.row;
        // while(v.pad-r>0)
        // {
        // v.colum.push_back(-1);
        // r++;
        // }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<float> demo(v.row*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = v.pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                if(v.row%wide==0){
                    demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                }else{
                    if(bId < (bIds-1))
                        demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                    else
                        demo[bId*window*wide + p*(v.row%wide) +bInId]=degree[j]*degree[column[m]];
                }
            }
            p++;
        }
        v.value=demo;
        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<float> valueNew;
    //按顺序整合res
    for (const auto& pair : res) {
     rowNew.push_back(rowNew.back()+pair.second.row);
    //  rowNew.push_back(rowNew.back()+pair.second.pad-pair.second.row);
     colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
     valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
    }

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat32);
    
    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    
    return {rowTensor,colTensor,valueTensor};
}





//balance
struct Value_balance_fp16 {
    std::vector<at::Half> value;
    std::vector<int> colum;
    std::vector<int> row;
    std::vector<int> window;
    std::vector<int> atomic;
};

struct Value_balance_tf32 {
    std::vector<float> value;
    std::vector<int> colum;
    std::vector<int> row;
    std::vector<int> window;
    std::vector<int> atomic;
};

struct Value_balance_sddmm {
    std::vector<int> value;
    std::vector<int> colum;
    std::vector<int> row;
    std::vector<int> window;
    std::vector<int> atomic;
};

std::vector<torch::Tensor> blockProcess_fp16_balance(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1, int partSize_t)
{
    // auto start = std::chrono::high_resolution_clock::now();
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<at::Half, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_balance_fp16> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_balance_fp16 v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        int v_size = mergedSet.size();
        int pad=((v_size/wide+1))*wide;
        if(v_size%wide==0)  pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        // int r=v_size;
        // while(pad-r>0)
        // {
        // v.colum.push_back(-1);
        // r++;
        // }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<at::Half> demo(v_size*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                // if(bId < (bIds-1))
                //     demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                // else
                //     demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];

                if(v_size%wide==0){
                    demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                }else{
                    if(bId < (bIds-1))
                        demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                    else
                        demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];
                }
            }
            p++;
        }
        v.value=demo;
        //开始load balance划分
        //计算有多少block
        int blocks = pad/wide; 
        if(blocks  > 0){
        if(blocks <= partSize_t)
        {
            v.row.push_back(v_size);
            // v.row.push_back(pad-v_size);
            v.window.push_back(i);
            v.atomic.push_back(0);
        }else{
             // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;

            if(block_residue > 0){
                for(int j=0; j<part_number; j++)
                {
                    v.row.push_back(partSize_t*wide);
                    // v.row.push_back(0);
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
                v.row.push_back(block_residue*wide - (pad-v_size));
                // v.row.push_back(pad - v_size);
                v.window.push_back(i);
                v.atomic.push_back(1);
            }else{
                for(int j=0; j<part_number; j++)
                {
                    //如果是最后一组block
                    if(j == (part_number-1)){
                        v.row.push_back((partSize_t*wide) - (pad-v_size));
                        // v.row.push_back(pad-v_size);
                    }else{
                        v.row.push_back(partSize_t*wide);
                        // v.row.push_back(0);
                    }
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
            }
        }
    }

        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<at::Half> valueNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    //按顺序整合res
    for (const auto& pair : res) {
        for(int sub : pair.second.row)
        {
            rowNew.push_back(rowNew.back()+sub);
        }  
        colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
        valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
        t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
        t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
    }
    // auto end = std::chrono::high_resolution_clock::now();

    // // 计算时间差并转换为毫秒
    // std::chrono::duration<double, std::milli> duration = end - start;
    // std::cout << duration.count() << " ms" << std::endl;
    
    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat16);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    
    return {rowTensor,colTensor,valueTensor,t_window_rowTensor,t_atomicTensor};
}


std::vector<torch::Tensor> blockProcess_tf32_balance(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1, int partSize_t)
{
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    auto degree=degree1.accessor<float, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_balance_tf32> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_balance_tf32 v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        int v_size = mergedSet.size();
        int pad=((v_size/wide+1))*wide;
        if(v_size%wide==0)  pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        // int r=v_size;
        // while(pad-r>0)
        // {
        // v.colum.push_back(-1);
        // r++;
        // }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<float> demo(v_size*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                // if(bId < (bIds-1))
                //     demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                // else
                //     demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];
                if(v_size%wide==0){
                    demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                }else{
                    if(bId < (bIds-1))
                        demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                    else
                        demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];
                }
            }
            p++;
        }
        v.value=demo;
        //开始load balance划分
        //计算有多少block
        int blocks = pad/wide; 
        if(blocks <= partSize_t)
        {
            v.row.push_back(v_size);
            // v.row.push_back(pad);
            v.window.push_back(i);
            v.atomic.push_back(0);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;

            if(block_residue > 0){
                for(int j=0; j<part_number; j++)
                {
                    v.row.push_back(partSize_t*wide);
                    // v.row.push_back(0);
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
                v.row.push_back(block_residue*wide - (pad-v_size));
                // v.row.push_back(pad - v_size);
                v.window.push_back(i);
                v.atomic.push_back(1);
            }else{
                for(int j=0; j<part_number; j++)
                {
                    //如果是最后一组block
                    if(j == (part_number-1)){
                        v.row.push_back((partSize_t*wide) - (pad-v_size));
                        // v.row.push_back(pad-v_size);
                    }else{
                        v.row.push_back(partSize_t*wide);
                        // v.row.push_back(0);
                    }
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
            }
        }

        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<float> valueNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    //按顺序整合res
    for (const auto& pair : res) {
        for(int sub : pair.second.row)
        {
            rowNew.push_back(rowNew.back()+sub);
        }  
        colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
        valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
        t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
        t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
    }

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kFloat32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    
    return {rowTensor,colTensor,valueTensor,t_window_rowTensor,t_atomicTensor};
}


//8x16划块
std::vector<torch::Tensor> blockProcess_sddmm(torch::Tensor row1, torch::Tensor column1, int window1, int wide1, int partSize_t)
{
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // auto degree=degree1.accessor<int, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_balance_sddmm> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_balance_sddmm v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        int v_size = mergedSet.size();
        int pad=((v_size/wide+1))*wide;
        if(v_size%wide==0)  pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        // int r=v_size;
        // while(pad-r>0)
        // {
        // v.colum.push_back(-1);
        // r++;
        // }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<int> demo(v_size*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                // if(bId < (bIds-1))
                //     demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                // else
                //     demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];
                if(v_size%wide==0){
                    demo[bId*window*wide + p*wide +bInId]=1;
                }else{
                    if(bId < (bIds-1))
                        demo[bId*window*wide + p*wide +bInId]=1;
                    else
                        demo[bId*window*wide + p*(v_size%wide) +bInId]=1;
                }
            }
            p++;
        }
        v.value=demo;
        //开始load balance划分
        //计算有多少block
        int blocks = pad/wide; 
        if(blocks <= partSize_t)
        {
            v.row.push_back(v_size);
            // v.row.push_back(pad);
            v.window.push_back(i);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;

            if(block_residue > 0){
                for(int j=0; j<part_number; j++)
                {
                    v.row.push_back(partSize_t*wide);
                    // v.row.push_back(0);
                    v.window.push_back(i);
                }
                v.row.push_back(block_residue*wide - (pad-v_size));
                // v.row.push_back(pad - v_size);
                v.window.push_back(i);
            }else{
                for(int j=0; j<part_number; j++)
                {
                    //如果是最后一组block
                    if(j == (part_number-1)){
                        v.row.push_back((partSize_t*wide) - (pad-v_size));
                        // v.row.push_back(pad-v_size);
                    }else{
                        v.row.push_back(partSize_t*wide);
                        // v.row.push_back(0);
                    }
                    v.window.push_back(i);
                }
            }
        }

        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<int> valueNew;
    std::vector<int> t_window_rowNew;
    //按顺序整合res
    for (const auto& pair : res) {
        for(int sub : pair.second.row)
        {
            rowNew.push_back(rowNew.back()+sub);
        }  
        colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
        valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
        t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
    }

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);

    
    return {rowTensor,colTensor,valueTensor,t_window_rowTensor};
}



//gnn
std::vector<torch::Tensor> blockProcess_sddmm_gnn(torch::Tensor row1, torch::Tensor column1, int window1, int wide1, int partSize_t)
{
    int window = window1;
    int wide = wide1;
    auto *row=row1.data_ptr<int>();
    auto column=column1.accessor<int, 1>();
    // auto degree=degree1.accessor<int, 1>();
    int rows=row1.size(0)-1;
    int rowsNew=rows/window;
    //最终的map
    std::map<int, Value_balance_sddmm> res;
    //按照rowoffset，每8行进行一次block的组合
    #pragma omp parallel for
    for(int i=0;i<rowsNew;i++)
    {
        Value_balance_sddmm v;
        //对column进行合并，找并集，并按照升序排列
        std::set<int> mergedSet;
        std::copy(&column[row[i*window]], &column[row[i*window+window]], std::inserter(mergedSet, mergedSet.end()));
        //column按wide划分，填充value值
        int v_size = mergedSet.size();
        int pad=((v_size/wide+1))*wide;
        if(v_size%wide==0)  pad-=wide;
        // 将 set 容器转化为 vector 容器，并按升序排列
        std::vector<int> mergedVector(mergedSet.begin(), mergedSet.end());
        //填充column值，padding的部分为-1
        v.colum=mergedVector;
        // int r=v_size;
        // while(pad-r>0)
        // {
        // v.colum.push_back(-1);
        // r++;
        // }
        //填充value值，padding的均为0
        //生成value的0模板，然后填充存在的value值
        std::vector<int> demo(v_size*window);
        //先创建一个列索引的map
        std::map<int, int> colmap;
        int c=0;
        for(auto col:mergedVector)
        colmap[col]=c++;
        //在有值的位置写入相应的value值
        int bIds = pad / wide;
        int p=0;
        for(int j=i*window;j<(i+1)*window;j++)
        {
            for(int m=row[j];m<row[j+1];m++)
            {
                //存储按8列为一个block存储
                int bId=colmap[column[m]]/wide;
                int bInId=colmap[column[m]]%wide;
                // if(bId < (bIds-1))
                //     demo[bId*window*wide + p*wide +bInId]=degree[j]*degree[column[m]];
                // else
                //     demo[bId*window*wide + p*(v_size%wide) +bInId]=degree[j]*degree[column[m]];
                if(v_size%wide==0){
                    demo[bId*window*wide + p*wide +bInId]=1;
                }else{
                    if(bId < (bIds-1))
                        demo[bId*window*wide + p*wide +bInId]=1;
                    else
                        demo[bId*window*wide + p*(v_size%wide) +bInId]=1;
                }
            }
            p++;
        }
        v.value=demo;
        //开始load balance划分
        //计算有多少block
        int blocks = pad/wide; 
        if(blocks <= partSize_t)
        {
            v.row.push_back(v_size);
            // v.row.push_back(pad);
            v.window.push_back(i);
            v.atomic.push_back(0);
        }else{
            // 对block太多的块进行分割
            int part_number = blocks / partSize_t;
            int block_residue = blocks % partSize_t;

            if(block_residue > 0){
                for(int j=0; j<part_number; j++)
                {
                    v.row.push_back(partSize_t*wide);
                    // v.row.push_back(0);
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
                v.row.push_back(block_residue*wide - (pad-v_size));
                // v.row.push_back(pad - v_size);
                v.window.push_back(i);
                v.atomic.push_back(1);
            }else{
                for(int j=0; j<part_number; j++)
                {
                    //如果是最后一组block
                    if(j == (part_number-1)){
                        v.row.push_back((partSize_t*wide) - (pad-v_size));
                        // v.row.push_back(pad-v_size);
                    }else{
                        v.row.push_back(partSize_t*wide);
                        // v.row.push_back(0);
                    }
                    v.window.push_back(i);
                    v.atomic.push_back(1);
                }
            }
        }

        //封装v
        #pragma omp critical  
        {  res[i]=v;}
    }
    std::vector<int> rowNew;
    rowNew.push_back(0);
    std::vector<int> colNew;
    std::vector<int> valueNew;
    std::vector<int> t_window_rowNew;
    std::vector<int> t_atomicNew;
    //按顺序整合res
    for (const auto& pair : res) {
        for(int sub : pair.second.row)
        {
            rowNew.push_back(rowNew.back()+sub);
        }  
        colNew.insert(colNew.end(),pair.second.colum.begin(),pair.second.colum.end());
        valueNew.insert(valueNew.end(),pair.second.value.begin(),pair.second.value.end());
        t_window_rowNew.insert(t_window_rowNew.end(),pair.second.window.begin(),pair.second.window.end());
        t_atomicNew.insert(t_atomicNew.end(),pair.second.atomic.begin(),pair.second.atomic.end());
    }

    auto rowTensor1 = torch::from_blob(rowNew.data(), rowNew.size(), torch::kInt32);
    auto colTensor1 = torch::from_blob(colNew.data(), colNew.size(), torch::kInt32);
    auto valueTensor1 = torch::from_blob(valueNew.data(), valueNew.size(), torch::kInt32);
    auto t_window_rowTensor1 = torch::from_blob(t_window_rowNew.data(), t_window_rowNew.size(), torch::kInt32);
    auto t_atomicTensor1 = torch::from_blob(t_atomicNew.data(), t_atomicNew.size(), torch::kInt32);

    torch::Tensor rowTensor = torch::empty_like(rowTensor1);
    rowTensor.copy_(rowTensor1);
    torch::Tensor colTensor = torch::empty_like(colTensor1);
    colTensor.copy_(colTensor1);
    torch::Tensor valueTensor = torch::empty_like(valueTensor1);
    valueTensor.copy_(valueTensor1);
    torch::Tensor t_window_rowTensor = torch::empty_like(t_window_rowTensor1);
    t_window_rowTensor.copy_(t_window_rowTensor1);
    torch::Tensor t_atomicTensor = torch::empty_like(t_atomicTensor1);
    t_atomicTensor.copy_(t_atomicTensor1);
    
    return {rowTensor,colTensor,valueTensor,t_window_rowTensor,t_atomicTensor};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    /*
    New----------
    */

    // block
    m.def("blockProcess_fp16", &blockProcess_fp16, "Block for FP16 with any shape");
    m.def("blockProcess_fp16_ori", &blockProcess_fp16_ori, "Block for FP16 with any shape");

    m.def("blockProcess_tf32", &blockProcess_tf32, "Block for TF32 with any shape");
    // + load balance
    m.def("blockProcess_fp16_balance", &blockProcess_fp16_balance, "Block for FP16 with any shape");
    m.def("blockProcess_tf32_balance", &blockProcess_tf32_balance, "Block for TF32 with any shape");

    // sddmm output templete,
    m.def("blockProcess_sddmm_balance", &blockProcess_sddmm, "SDDMM with output 8x16 and save as 8x4 or 8x8");
    m.def("blockProcess_sddmm_balance_gnn", &blockProcess_sddmm_gnn, "SDDMM with output 8x16 and save as 8x4 or 8x8");


    // GAT trans using CSR,  return 8x16, 8x8, 8x4
    m.def("blockProcess_csr", &blockProcess8_16_csr, "SDDMM for FP16 and TF32 with output 8x16 with CSR");
}