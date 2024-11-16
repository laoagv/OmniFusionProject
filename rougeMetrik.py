# -*- coding: utf-8 -*-
from rouge import Rouge 
rouge = Rouge()
count=0
hypothesis = {}
for i in range(32):
	hypothesis[i]={}
reference = "На изображении представлены различные компоненты и связи в облачной инфраструктуре. Давайте выделим все объекты и перечислим их, а также определим потенциальные уязвимости.\nОбъекты на изображении:\nStorage Account (аккаунты хранилища) - Серые и оранжевые значки с иконкой базы данных.\nVirtual Machine (виртуальные машины) - Серые и оранжевые значки с иконкой монитора.\nPublic IP (публичный IP) - Оранжевые значки с иконкой местоположения.\nSQL Server (сервер базы данных SQL) - Значок SQL сервера.\nFunction App - Значки, представляющие приложение функции (Function App).\nApp Service - Значки, представляющие сервис приложений (App Service).\nIdentity (идентификация) - Иконки с силуэтом пользователя.\nNetwork Service Group - Значки, представляющие группы сетевых сервисов.\nSensitive Data (конфиденциальные данные) - Обозначены как Sensitive Data.\nSE-DEV-081 - Основной кластер или узел, к которому подключены многие сервисы.\nПотенциальные уязвимости:\nИнтернет-доступ к виртуальным машинам и SQL Server:\nНа изображении несколько объектов помечены как Internet Exposed (подключены к Интернету), что обозначено оранжевыми значками и текстом.\nДоступ к SQL Server через публичный IP может привести к атакам, включая SQL-инъекции и брутфорс-атаки на учетные данные.\nВиртуальные машины, подключенные к Интернету, также могут быть уязвимы к несанкционированному доступу и различным сетевым атакам.\nКонфиденциальные данные (Sensitive Data):\nХранение конфиденциальных данных в доступном из Интернета хранилище представляет значительный риск утечки данных.\nПри компрометации идентификационных данных можно получить доступ к этому хранилищу.\nОтсутствие явной сегментации сети:\nЕсли между объектами нет достаточной сегментации (например, изолированных виртуальных сетей и ограничений на уровне сетевых групп безопасности), злоумышленник, получивший доступ к одному компоненту, может быстро перемещаться по системе.\nИспользование Public IP и Identity без дополнительных мер безопасности:\nПубличные IP-адреса предоставляют точки входа для атак, особенно если отсутствуют дополнительные меры, такие как VPN, брандмауэры, правила сетевого контроля доступа (ACL) и двухфакторная аутентификация для аккаунтов с критичным доступом.\nОтсутствие контроля за управлением идентификацией (Identity), особенно если оно базируется на слабых политиках безопасности, может позволить злоумышленнику использовать украденные учетные данные для получения доступа.\nПриложения и сервисы с неограниченным доступом к хранилищу:\nУ приложений (Function App и App Service) и виртуальных машин имеется прямой доступ к Storage Account и SQL Server. Без соответствующего контроля доступности, это создает риск компрометации данных при захвате управления над одним из сервисов.\nОтсутствие упоминания шифрования и других мер защиты данных:\nНет признаков того, что конфиденциальные данные шифруются в хранилище или во время передачи. Это может подвергнуть данные риску утечки.\nРекомендации по безопасности:\nОграничить интернет-доступ к критическим компонентам (например, к SQL Server и виртуальным машинам) и настроить доступ через VPN или защищенные каналы.\nСегментировать сеть и использовать группы сетевых сервисов, чтобы ограничить перемещение внутри сети.\nВключить многофакторную аутентификацию (MFA) и строгое управление идентификацией для доступа к важным сервисам.\nЗашифровать конфиденциальные данные как на уровне хранения, так и на уровне передачи данных.\nНастроить брандмауэры и правила доступа к IP для управления доступом на основе ролей.\nРегулярно проводить аудит и мониторинг безопасности для отслеживания подозрительной активности и своевременного устранения уязвимостей.\nЭти меры помогут уменьшить риски, связанные с этой инфраструктурой."
with open("logsPrompts.txt","r",encoding="utf-8") as f:
	for line in f:
		if "PROMPT:" in line:
			hypothesis[count]["prompt"]=line
		if "do_sample:" in line:
			hypothesis[count]["params"]=line
			# print(line)
		if "Start time:" in line:
			hypothesis[count]["time"]=line
			# print(line)
		if "Answer:  " in line:

			scores = rouge.get_scores(line, reference)
			hypothesis[count]["metrik"]=scores
			hypothesis[count]["answer"]=line
			count+=1
			# print(scores)

listTuples=[]

for j in range(32):
	listTuples.append((j, hypothesis[j]["metrik"][0]["rouge-l"]))
	# print(hypothesis[j]["metrik"])
sortedListTuples = sorted(listTuples, key=lambda x:x[1]["f"])
print(sortedListTuples)


print(hypothesis[10]["prompt"])
print(hypothesis[10]["params"])
print(hypothesis[10]["time"])
print(hypothesis[10]["answer"])
print(hypothesis[10]["metrik"])


print(hypothesis[27]["prompt"])
print(hypothesis[27]["params"])
print(hypothesis[27]["time"])
print(hypothesis[27]["answer"])
print(hypothesis[27]["metrik"])

print(hypothesis[29]["prompt"])
print(hypothesis[29]["params"])
print(hypothesis[29]["time"])
print(hypothesis[29]["answer"])
print(hypothesis[29]["metrik"])

print(hypothesis[25]["prompt"])
print(hypothesis[25]["params"])
print(hypothesis[25]["time"])
print(hypothesis[25]["answer"])
print(hypothesis[25]["metrik"])




print(count)
