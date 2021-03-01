#Metodo Monte carlo para estimar o valor numeral de PI'
from __future__ import print_function, division
import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def estimatePi(n, block=100000):
    #block = n de iteracoes

    # Retorna uma estimativa de pi atraves da insercao de N numeros randomicos em 
    # um quadrado [[-1,1]],[-1,1]] e calcula qual fracao fica dentro do circulo unitario

    # Desenha numeros aleatorios em blocos do tamanho especificado, mantem a contagem
    # dos numeros de pontos dentro do circulo

    total_number = 0
    i = 0
    while i < n:
        if n-i < block:
            block = n-i
        points = genPoints(block)
        number_in_circle = np.sum(inCircle(points))
        total_number += number_in_circle
        i += block
    return (4.*total_number)/n

def estimatePiParallel(comm, N):
    # Em execucao dos processos disponiveis, eh calculado uma
    # estimativa do PI, removendo N numeros aleatorios 
    # e caculado uma estimativa do pi

    if rank == 0:
        data = [N for i in range(size)]
    else:
        data = None
    #Scatter = envia todos os dados para todos os processos de um comunicador
    data = comm.scatter(data, root=0)

    #Busca as estimativas do PI pelo numero de termos
    piEst = estimatePi(N)

    #Gather = Pega os elementos de muitos processos de comunicadores, e reune em um sp
    pi_estimates = comm.gather(piEst, root=0)

    # o processo de gerenciamento ira reunir todas as estimativas
    # produzido por todos os trabalhadores, e calcular a media e
    # desvio padrao entre as execucoes independentes
   
    if rank == 0:
        return pi_estimates


def estimatePiStats(comm, n_executions, n_executions_workers):
    results = []
    # Executado a estimativas do PI paralelas de acordo com o numero
    # de execucoes de cada trabalhador, sendo anexada seus resultados quando o rank
    # de execucao estiver = 0, retornando a media e desvio padrao para a funcao main
    for i in range(n_executions_workers):
        result = estimatePiParallel(comm, n_executions)
        if rank == 0:
            results.append(result)
    if rank == 0:
        pi_avg_est = np.mean(results)
        pi_std_est  = np.std(results)
        return pi_avg_est, pi_std_est

def genPoints(n):
    # Retorna uma matriz uniforme de N pares randomicos (x,y) dentro
    # do quadrado qual o circulo esta centralizado
    # Quadrado com cantos em (-1,-1), (-1,1), (1,1), (1,-1)
    points = 2 * np.random.random((n,2)) - 1
    return points

def inCircle(p):
    
    # Retorna um array booleando, passando True/False os elementos que correspondem 
    # aos pontos dentro do circulo, dentro do quadrado. 
    # np.linlang.norm = encontrar tamanho do vetor
    return np.linalg.norm(p,axis=-1)<=1.0

def printToFile(estimates):
    plt.figure()
    plt.errorbar(np.log2(estimates[:,0]), estimates[:,1], yerr=estimates[:,2])
    plt.ylabel('Estimativa numero PI')
    plt.xlabel('Tentativas N (log2)')
    plt.savefig('avg-PI-vs-Trys.png')

    #
    plt.figure()
    plt.ylabel('Desvio Padrao PI (log2)')
    plt.xlabel('Tentativas N (log2)')
    plt.plot(np.log2(estimates[:,0]), np.log2(estimates[:,2]))
    plt.savefig('std-PI-vs-Trys.png')


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    #Balancear execucoes por comunicadores
    executions_total = 64 # 64 execucoes independentes para cada N 
    executions_per_workers = executions_total // size
    
    if rank == 0:
        #Evitar dados gravados no buffer do terminal
        sys.stdout.flush()

    estimates = []
    for trys in range(5,15):
    # Gerar randomicos  4 * (5-14) potencias inteiras de 4, calcular media e desvio
    # padrao das estimativas de pi, por tentativas de execucoes totais
    # distribuido por todos os trabalhadores / comunicadores
        N = int(4**trys)
        result = estimatePiStats(comm, N, executions_per_workers)
        if rank == 0:
            pi_avg_est, pi_std_est = result
            estimates.append((N, pi_avg_est, pi_std_est))
            print('N aleatorios tentados: %s | Media estimada do PI: %s | Desvio padrao estimado do PI para estas tentativas: %s' % (N, pi_avg_est, pi_std_est) )
            #Evitar dados gravados no buffer do terminal
            sys.stdout.flush()

    if rank == 0:
        printToFile(np.array(estimates))
        
       
    MPI.Finalize()