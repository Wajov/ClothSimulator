#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <tuple>

const int N = 500;
int indices[N][N];

int main() {
    // std::ifstream fin;

    // double x;
    // fin.open("standard_a.txt");
    // std::vector<double> a;
    // while (fin >> x)
    //     a.push_back(x);
    // fin.close();
    // fin.open("standard_b.txt");
    // std::vector<double> b;
    // while (fin >> x)
    //     b.push_back(x);
    // fin.close();

    // fin.open("output_a.txt");
    // std::vector<double> c;
    // while (fin >> x)
    //     c.push_back(x);
    // fin.close();
    // fin.open("output_b.txt");
    // std::vector<double> d;
    // while (fin >> x)
    //     d.push_back(x);
    // fin.close();

    // int N = b.size();

    // for (int i = 0; i < N * N; i++) {
    //     double diff = std::abs(a[i] - c[i]);
    //     if (diff > 1e-5)
    //         std::cout << "A(" << i / N << ", " << i % N << "): " << a[i] << ' ' << c[i] << ' ' << diff << std::endl;
    // }
    // for (int i = 0; i < N; i++) {
    //     double diff = std::abs(b[i] - d[i]);
    //     if (diff > 1e-5)
    //         std::cout << "b(" << i << "): " << b[i] << ' ' << d[i] << ' ' << diff << std::endl;
    // }

    // int v0, v1, v2, v3;

    // fin.open("standard_impacts.txt");
    // std::set<std::tuple<int, int, int, int>> s;
    // while (fin >> v0 >> v1 >> v2 >> v3) {
    //     std::vector<int> tmp = {v0, v1, v2, v3};
    //     std::sort(tmp.begin(), tmp.end());
    //     s.insert(std::make_tuple(tmp[0], tmp[1], tmp[2], tmp[3]));
    // }
    // fin.close();

    // fin.open("output_impacts.txt");
    // std::set<std::tuple<int, int, int, int>> t;
    // while (fin >> v0 >> v1 >> v2 >> v3) {
    //     std::vector<int> tmp = {v0, v1, v2, v3};
    //     std::sort(tmp.begin(), tmp.end());
    //     t.insert(std::make_tuple(tmp[0], tmp[1], tmp[2], tmp[3]));
    // }
    // fin.close();

    // std::cout << s.size() << ' ' << t.size() << std::endl;
    // for (auto& tu : s)
    //     if (t.find(tu) == t.end())
    //         std::cout << "0: " << std::get<0>(tu) << ' ' << std::get<1>(tu) << ' ' << std::get<2>(tu) << ' ' << std::get<3>(tu) << ' ' << std::endl;
    
    // for (auto& tu : t)
    //     if (s.find(tu) == s.end())
    //         std::cout << "1: " << std::get<0>(tu) << ' ' << std::get<1>(tu) << ' ' << std::get<2>(tu) << ' ' << std::get<3>(tu) << ' ' << std::endl;

    // double y, z, w;

    // fin.open("standard_post.txt");
    // std::vector<std::vector<double>> f;
    // while (fin >> x >> y >> z)
    //     f.push_back({x, y, z});
    // fin.close();

    // fin.open("output_post.txt");
    // std::vector<std::vector<double>> g;
    // while (fin >> x >> y >> z)
    //     g.push_back({x, y, z});
    // fin.close();

    // for (int i = 0; i < f.size(); i++)
    //     for (int j = 0; j < 3; j++) {
    //         double diff0 = std::abs(f[i][j] - g[i][j]);
    //         double diff1 = std::abs((f[i][j] - g[i][j]) / f[i][j]);
    //         if (diff1 > 1e-2)
    //             std::cout << i << ' ' << j << ' ' << f[i][j] << ' ' << g[i][j] << ' ' << diff1 << std::endl;
    //     }

    // double x, y, z, w;
    // fin.open("standard_sizing.txt");
    // std::vector<std::vector<double>> a;
    // while (fin >> x >> y >> z >> w)
    //     a.push_back({x, y, z, w});
    // fin.close();

    // fin.open("output_sizing.txt");
    // std::vector<std::vector<double>> b;
    // while (fin >> x >> y >> z >> w)
    //     b.push_back({x, y, z, w});
    // fin.close();

    // for (int i = 0; i < a.size(); i++)
    //     for (int j = 0; j < 4; j++) {
    //         double diff0 = std::abs(a[i][j] - b[i][j]);
    //         double diff1 = std::abs((a[i][j] - b[i][j]) / a[i][j]);
    //         if (std::min(diff0, diff1) > 1e-2)
    //             std::cout << i << ' ' << j << ' ' << a[i][j] << ' ' << b[i][j] << ' ' << std::min(diff0, diff1) << std::endl;
    //     }
    
    std::ofstream fout("meshes/test.obj");
    float d = 2.0f / (N - 1);
    int num = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            indices[i][j] = ++num;
            float x = -1.0f + i * d;
            float y = -1.0f + j * d;
            fout << "v " << x << ' ' << y << ' ' << 0 << std::endl;
            fout << "vt " << x << ' ' << y << std::endl;
        }

    for (int i = 0; i < N - 1; i++)
        for (int j = 0; j < N - 1; j++) {
            fout << "f " << indices[i][j] << '/' << indices[i][j] << ' ' << indices[i][j + 1] << '/' << indices[i][j + 1] << ' ' << indices[i + 1][j] << '/' << indices[i + 1][j] << std::endl;
            fout << "f " << indices[i + 1][j + 1] << '/' << indices[i + 1][j + 1] << ' ' << indices[i + 1][j] << '/' << indices[i + 1][j] << ' ' << indices[i][j + 1] << '/' << indices[i][j + 1] << std::endl;
        }

    return 0;
}