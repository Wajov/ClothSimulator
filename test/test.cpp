#include <iostream>
#include <fstream>

int main() {
    std::ifstream fin;

    fin.open("standard.txt");
    double a[244][243];
    for (int i = 0; i < 244; i++)
        for (int j = 0; j < 243; j++)
            fin >> a[i][j];
    fin.close();

    fin.open("output.txt");
    double b[244][243];
    for (int i = 0; i < 244; i++)
        for (int j = 0; j < 243; j++)
            fin >> b[i][j];
    fin.close();

    for (int i = 0; i < 244; i++)
        for (int j = 0; j < 243; j++) {
            double diff = std::abs(a[i][j] - b[i][j]);
            if (diff > 1e-7)
                std::cout << i << ' ' << j << ' ' << a[i][j] << ' ' << b[i][j] << ' ' << diff << std::endl;
        }

    return 0;
}