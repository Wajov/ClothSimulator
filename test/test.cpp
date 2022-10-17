#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <tuple>

int main() {
    std::ifstream fin;

    // fin.open("standard.txt");
    // double a[244][243];
    // for (int i = 0; i < 244; i++)
    //     for (int j = 0; j < 243; j++)
    //         fin >> a[i][j];
    // fin.close();

    // fin.open("output.txt");
    // double b[244][243];
    // for (int i = 0; i < 244; i++)
    //     for (int j = 0; j < 243; j++)
    //         fin >> b[i][j];
    // fin.close();

    // for (int i = 0; i < 244; i++)
    //     for (int j = 0; j < 243; j++) {
    //         double diff = std::abs(a[i][j] - b[i][j]);
    //         if (diff > 1e-6)
    //             std::cout << i << ' ' << j << ' ' << a[i][j] << ' ' << b[i][j] << ' ' << diff << std::endl;
    //     }

    int v0, v1, v2, v3;

    fin.open("standard_impacts.txt");
    std::set<std::tuple<int, int, int, int>> s;
    while (fin >> v0 >> v1 >> v2 >> v3) {
        std::vector<int> tmp = {v0, v1, v2, v3};
        std::sort(tmp.begin(), tmp.end());
        s.insert(std::make_tuple(tmp[0], tmp[1], tmp[2], tmp[3]));
    }
    fin.close();

    fin.open("output_impacts.txt");
    std::set<std::tuple<int, int, int, int>> t;
    while (fin >> v0 >> v1 >> v2 >> v3) {
        std::vector<int> tmp = {v0, v1, v2, v3};
        std::sort(tmp.begin(), tmp.end());
        t.insert(std::make_tuple(tmp[0], tmp[1], tmp[2], tmp[3]));
    }
    fin.close();

    for (auto& tu : s)
        if (t.find(tu) == t.end())
            std::cout << "0: " << std::get<0>(tu) << ' ' << std::get<1>(tu) << ' ' << std::get<2>(tu) << ' ' << std::get<3>(tu) << ' ' << std::endl;
    
    for (auto& tu : t)
        if (s.find(tu) == s.end())
            std::cout << "1: " << std::get<0>(tu) << ' ' << std::get<1>(tu) << ' ' << std::get<2>(tu) << ' ' << std::get<3>(tu) << ' ' << std::endl;

    return 0;
}