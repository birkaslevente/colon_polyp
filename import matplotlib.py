import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('erinto.csv', header=None, names=['N', 'Hiba'])

# Adatok logaritmikus skálára konvertálása
log_N = np.log10(data['N'])
log_Hiba = np.log10(data['Hiba'])

# Lineáris regresszió a logaritmikus adatokra
coeffs = np.polyfit(log_N, log_Hiba, 1)
m, b = coeffs

# Egyenes egyenletének kiszámítása
fit_line = m * log_N + b

# Ábra készítése
plt.figure(figsize=(10, 6))
plt.scatter(log_N, log_Hiba, color='blue', marker='o', label='Mérési pontok')
plt.plot(log_N, fit_line, color='red', linestyle='-', label=f'Illesztett egyenes (m={m:.4f})')

plt.title('Hiba a mintaszám függvényében (log-log skála)')
plt.xlabel('log(N)')
plt.ylabel('log(Hiba)')
plt.grid(True, alpha=0.3)
plt.legend()

# Eredmények kiírása
print(f"Az illesztett egyenes meredeksége: {m:.4f}")
print(f"Az illesztett egyenes egyenlete: log(Hiba) = {m:.4f} * log(N) + {b:.4f}")

plt.savefig('erinto_hiba.png', dpi=300)
plt.show()
