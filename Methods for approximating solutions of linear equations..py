import copy
import math

# am folosit python 3.8

# Ex 1

# functia verifica daca dimensiunile sunt potrivite (se potrivesc intre ele) si daca matricea este patratica
def validare(A, b):
    for i in range(len(A)):
        assert len(A) == len(A[i]), 'Matricea sistemului nu este patratica!'
    assert len(b) == len(A), 'Matricea sistemului si b nu se potrivesc!'

# functia gaseste valoarea maxima (valoarea maxima in modul, cea mai departata de 0) din matrice, tinand cont de parametrii precizati p si m (am grija sa caut doar in partea de matrice delimitata)
def gasire_max(A, p, m, n):
    maxim = 0
    i = 0
    paux = p
    maux = m
    for x in A:
        j = 0
        for y in x:
            ok = 0
            if i >= n or j >= n or i < p or j < m:
                j = j + 1
                ok = 1
            elif abs(y) > abs(maxim):
                maxim = y
                paux = i
                maux = j
            if ok == 0:
                j = j + 1
        i = i + 1
    return paux, maux, maxim


# functie ajutatoare de calculare a sumei din cadrul valorii lui xk din metoda substitutiei descendente
def suma(A, x, k, n):
    s = 0
    for j in range(k + 1, n):
        s = s + A[k][j] * x[j]
    return s

def metoda_substitutiei_descendente(A, n):
    # 1. initializarea vectorului de solutie
    x = [0 for i in range(n)]
    x[n - 1] = 1 / A[n - 1][n - 1] * A[n - 1][n]
    k = n - 2
    # 2. aflarea solutiei numerice
    while k >= 0:
        x[k] = 1 / A[k][k] * (A[k][n] - suma(A, x, k, n))
        k = k - 1
    return x

def gauss_pivotare_totala(A, b, exercitiu):
    validare(A, b)
    n = len(A)
    index = [i + 1 for i in range(n)]
    i = 0

    # 1.construim matricea extinsa
    for x in A:
        x.append(b[i])
        i += 1

    # 2.Determinam unde este pivotul pe coloana k
    for k in range(n):
        # determinam maximul si primii indici p, m
        p, m, maxim = gasire_max(A, k, k, n)
        if maxim == 0:
            print("Sist. incomp. sau comp. nedet.")
            return
        if p != k:
            A[p], A[k] = A[k], A[p]
            # interschimbam liniile
        if m != k:
            # schimbam indicii nec.
            index[m], index[k] = index[k], index[m]
            for x in A:
                x[m], x[k] = x[k], x[m]
                # schimbam coloanele


        for l in range(k + 1, n):
            # construim 0 sub pozitia pivotului
            mlk = float(A[l][k] / A[k][k])
            A[l] = [A[l][aux] - mlk * A[k][aux] for aux in range(n + 1)]

    # 3.
    if A[n - 1][n - 1] == 0:
        print("Sist. incomp. sau comp. nedet.")
        return

    # 4.
    sol = [0 for i in range(n)]
    x = metoda_substitutiei_descendente(A, n)

    # in functie de parametrul exercitiu intors sau afisez rezultatele dupa cum am nevoie la ex 1 sau ex 2
    if exercitiu == 1:
        print('Solutia in funcite de indexi si nerotunjita: ', x, 'ordine indexi: ', index)

        ok = 0
        for i in index:
            sol[i - 1] = round(x[ok])
            ok = ok + 1
        print("Solutia rotunjita si ordonata: ", sol)
    else:
        ok = 0
        for i in index:
            sol[i - 1] = x[ok]
            ok = ok + 1
        return sol, index

# datele de intrare
a = [[0, -5, -3, 8], [0, -8, -4, 8], [-7, -1, -10, -4], [2, -8, 5, -1]]
b = [13, -12, -143, -11]
print('Ex. 1')
gauss_pivotare_totala(a, b, 1)
print('-' * 140)

# Ex 2

# datele de intrare
B = [[0, 4, 8, 4], [-3, 2, -5, -7], [1, 8, 8, -7], [-2, 6, 8, 6]]
I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

def inversa_gauss(A, I):
    # codul este destul de asemanator cu exercitiul 1
    n = len(A)
    index = [i + 1 for i in range(n)]

    # 2.
    for k in range(n):
        p, m, maxim = gasire_max(A, k, k, n)
        if maxim == 0:
            print("Sist. incomp. sau comp. nedet.")
            return
        if p != k:
            A[p], A[k] = A[k], A[p]
            I[p], I[k] = I[k], I[p]
            # aici tinem cont si de schimbarile de linii in matricea de identitate
        if m != k:
            index[m], index[k] = index[k], index[m]
            for x in A:
                x[m], x[k] = x[k], x[m]
            for x in I:
                x[m], x[k] = x[k], x[m]
            # aici tinem cont si de schimbarile de coloane in matricea de identitate

        for l in range(k + 1, n):
            llk = float(A[l][k] / A[k][k])
            A[l] = [A[l][aux] - llk * A[k][aux] for aux in range(n)]
            I[l] = [I[l][aux] - llk * I[k][aux] for aux in range(n)]

    # folosesc functtia gauss_pivotare_totala folosit si la ex 1 pentru a gasi solutiile pe rand pe coloane
    linie = [0 for i in range(n)]
    sol = [[], [], [], []]
    # practic am aceasi problema ca la exercitiul 1, doar ca acum am 4 sisteme si solutiile pentru fiecare dintre ele sunt in I, iar necunoscutele sunt coloanele din matricea inversa
    for i in range(n):
        aux = copy.deepcopy(A)
        for j in range(n):
            linie[j] = I[j][i]
        sol[index[i]-1], index2 = gauss_pivotare_totala(aux, linie, 2)
    # aici aduc solutiile la forma lor ordonata
    sol2 = copy.deepcopy(sol)
    aux = [x for x in range(n)]
    j = 0
    for linie in sol:
        i = 0

        for x in linie:
            aux[index[i]-1] = x
            i = i + 1

        for k in range(n):
            sol2[k][j] = aux[k]

        j = j + 1
    for x in sol2:
        print(x, '\n')


print('Ex. 2\nMatricea inversa este: \n')
inversa_gauss(B, I)
print('-'*140)

# Ex 3
print('Ex. 3\n')
# datele de intrare
a = [[0, -7, -2, -8], [-8, -7, -7, 4], [5, -9, -3, -7], [-1, -5, -6, -3]]
b = [-103, -81, -92, -86]

# asemanatoare cu functia gasire_max doar ca acum caut doar sub indicele de linie respectiv
def gasire_max2(A, p, m, n):
    maxim = 0
    i = 0
    paux = p
    maux = m
    for x in A:
        j = 0
        for y in x:
            ok = 0
            if i >= n or j >= n or i < p or j != m:
                j = j + 1
                ok = 1
            elif abs(y) > abs(maxim):
                maxim = y
                paux = i
                maux = j
            if ok == 0:
                j = j + 1
        i = i + 1
    return paux, maux, maxim

def factorizare_LU_GPP(A, b):
    validare(A, b)
    n = len(A)
    # 1.
    L = [[0 for x in range(n)]
         for y in range(n)]
    # construim matricea identitate de n
    for i in range(n):
        for j in range(n):
            if i == j:
                L[i][j] = 1
            else:
                L[i][j] = 0

    w = [i + 1 for i in range(n)]

    index = [i + 1 for i in range(n)]
    i = 0

    # 2.
    for k in range(n):
        # determinam maximul si primii indici p, m
        p, m, maxim = gasire_max2(A, k, k, n)

        if maxim == 0:
            print('Matricea nu admite factorizare LU')
            return

        if p != k:
            A[p], A[k] = A[k], A[p]
            w[p], w[k] = w[k], w[p]
            index[p], index[k] = index[k], index[p]
            # interschimbam

        if k > 1:
            for x in range(k-1):
                aux1 = L[l][x]
                L[l][x] = L[k][x]
                L[k][x] = aux1


        for l in range(k + 1, n):
            L[l][k] = float(A[l][k] / A[k][k])
            llk = L[l][k]
            A[l] = [A[l][aux] - llk * A[k][aux] for aux in range(n)]

    # 3.
    if A[n - 1][n - 1] == 0:
        print("Matricea nu admite factorizare LU")
        return

    # schimb intre ele liniile (1,4) si (2,3)

    L[0], L[3] = L[3], L[0]
    L[1], L[2] = L[2], L[1]

    # schimb intre ele coloanele (1,4) si (2,3), am facut aceste schimbari pentru a putea folosi metoda_substitutiei_descendente
    for line in L:
        line[0], line[3] = line[3], line[0]
        line[1], line[2] = line[2], line[1]

    # adaug b[i] la finalul fiecarei linii din L
    i = 0
    for line in L:
        line.append(b[i])
        i += 1

    aux = metoda_substitutiei_descendente(L, n)

    # adaug aux[i] la finalul fiecarei linii din A
    i = 0
    for line in A:
        line.append(aux[i])
        i += 1
    aux = metoda_substitutiei_descendente(A, n)

    # rearanjarea in ordine a solutiilor

    sol = [0 for i in range(n)]
    for i in range(4):
        sol[index[i]-4] = aux[i]
    print(sol)


factorizare_LU_GPP(a, b)

print('-'*140)

# Ex 4
print('Ex. 4\n')

# Matricea simetrica A ∈ Mn(R) este pozitiv definita
# daca si numai daca toti minorii principali, i.e. det Ak > 0, Ak = (aij) i,j=1,k

#Fie A ∈ Mn(R) simetrica si pozitiv definita. Daca componentele lkk , k = 1,n de pe diagonala principala a matricei L
# din descompunerea Cholesky sunt strict pozitive, atunci descompunerea este unica.

def descompunerea_Cholesky(C):
    n = len(C)
    sol = [[0 for x in range(n + 1)]
             for y in range(n + 1)]
    # 1.
    if C[0][0] <= 0:
        print('Matricea nu este pozitiv definita')
        return

    # 2.
    # descompunem matricea in triangulari inferioare
    for i in range(n):
        for j in range(i + 1):
            sum1 = 0

            # suma pentru diagonale
            if (j == i):
                for k in range(j):
                    sum1 += pow(sol[j][k], 2)
                if C[j][j] - sum1 <= 0:
                    print('Matricea nu este pozitiv definita')
                    return
                sol[j][j] = int(math.sqrt(C[j][j] - sum1))

            else:

                # Evaluam L(i, j) folosind L(j, j)
                for k in range(j):
                    sum1 += (sol[i][k] * sol[j][k])
                if (sol[j][j] > 0):
                    sol[i][j] = int((C[i][j] - sum1) /
                                      sol[j][j])

    # afisarea triangularii inferioare si a transpusei ei
    print("Triang. inferioara\t\t   Transpusa")
    for i in range(n):

        # Triangulare inferioara
        for j in range(n):
            print(sol[i][j], end="\t")
        print(end="\t     ")

        # Transpusa
        for j in range(n):
            print(sol[j][i], end="\t")
        print("")


# datele de intrare
C = [[9, -18, 21, -27], [-18, 37, -49, 56], [21, -49, 179, -113], [-27, 56, -113, 165]]
descompunerea_Cholesky(C)

print('-'*140)
