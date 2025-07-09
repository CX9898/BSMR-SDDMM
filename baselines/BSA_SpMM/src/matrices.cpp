#include "matrices.h"

void CSR::read_from_mtx(std::ifstream &fin, Option option, bool zero_base)
{
    string line;
    while (getline(fin, line))
    {
        if (line[0] == '%')
            continue;
        else
        {
            stringstream sin_meta(line);
            sin_meta >> rows >> cols >> total_nonzeros;
            break;
        }
    }
    printf("total_nonzeros: %d\n", total_nonzeros);
    map<intT, vector<intT>> dict;
    map<intT, vector<DataT>> vals;

    intT tnz = 0;
    while (getline(fin, line))
    {
        intT r, c;
        DataT v = 1;
        stringstream sin(line);
        if (pattern_only)
            sin >> r >> c;
        else
            sin >> r >> c >> v;

        if (not zero_base)
        {
            r--;
            c--;
        }

        if (dict.find(r) == dict.end())
        {
            dict[r] = vector<intT>();
            vals[r] = vector<DataT>();
        }
        dict[r].push_back(c);
        if (not pattern_only)
            vals[r].push_back(v);
        tnz++;
    }

    assert(tnz == total_nonzeros);
    // zero padding
    original_rows = rows;
    original_cols = cols;

    if (option.zero_padding)
    {
        if (rows % option.block_size)
        {
            original_rows = rows;
            rows = ((rows - 1) / option.block_size + 1) * option.block_size;
        }

        if (cols % option.block_size)
        {
            original_cols = cols;
            cols = ((cols - 1) / option.block_size + 1) * option.block_size;
        }
    }
    rowptr = new intT[rows + 1];
    // nzcount = new intT[rows];
    colidx = new intT[total_nonzeros];
    rowptr[0] = 0;
    if (not pattern_only)
    {
        values = new DataT[total_nonzeros];
    }
    intT offset = 0;
    for (int i = 0; i < rows; i++)
    {
        auto row_pos = dict[i];
        rowptr[i + 1] = rowptr[i] + row_pos.size();
        std::copy(dict[i].begin(), dict[i].end(), colidx + offset);

        if (not pattern_only)
        {
            std::copy(vals[i].begin(), vals[i].end(), values + offset);
        }
        offset += row_pos.size();
    }
}

void CSR::read_from_smtx(std::ifstream &fin, Option option, bool zero_base)
{
    string line;
    string buffer;
    map<intT, vector<intT>> dict;
    // header
    getline(fin, line);
    stringstream sin_meta(line);
    getline(sin_meta, buffer, ',');
    rows = stoi(buffer);
    getline(sin_meta, buffer, ',');
    cols = stoi(buffer);
    getline(sin_meta, buffer, ',');
    total_nonzeros = stoi(buffer);
    // zero padding
    original_rows = rows;
    original_cols = cols;
    if (option.zero_padding)
    {
        if (rows % option.block_size)
        {
            original_rows = rows;
            rows = ((rows - 1) / option.block_size + 1) * option.block_size;
        }
        if (cols % option.block_size)
        {
            original_cols = cols;
            cols = ((cols - 1) / option.block_size + 1) * option.block_size;
        }
    }
    rowptr = new intT[rows + 1];
    colidx = new intT[total_nonzeros];
    if (not pattern_only)
    {
        values = new DataT[total_nonzeros];
    }
    vector<intT> vec_colidx(total_nonzeros);
    // original_ja_contiguous = new intT[total_nonzeros];
    // vec_rowptr
    getline(fin, line);
    stringstream sin_row(line);
    intT offset = 0, idx = 0;
    while (getline(sin_row, buffer, ' '))
    {
        offset = stoi(buffer);
        rowptr[idx++] = offset;
        // vec_rowptr[idx++] = offset;
    }
    assert(idx == original_rows + 1);
    if (option.zero_padding)
    {
        for (int i = original_rows + 1; i <= rows; i++)
        {
            rowptr[i] = offset;
        }
    }
    int c;
    idx = 0;
    offset = 0;
    if (not pattern_only)
    {
        for (int i = 0; i < total_nonzeros; i++)
        {
            values[i] = 1;
        }
    }
    while (getline(fin, line))
    {
        stringstream sin_col(line);
        while (getline(sin_col, buffer, ' '))
        {
            c = stoi(buffer);
            vec_colidx[offset++] = c;
        }
    }

    std::copy(vec_colidx.begin(), vec_colidx.end(), colidx);
    assert(offset == total_nonzeros);
    vec_colidx.clear();
}