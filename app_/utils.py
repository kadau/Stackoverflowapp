# Import modules

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.externals import joblib
import pickle

# Most important Tags with more frequency. It will be used for supervised models

tag_s=['.net', 'actionscript-3', 'ajax', 'algorithm', 'apache','arrays', 'asp.net', 'asp.net-mvc',\
'c', 'c#', 'c++', 'cocoa','cocoa-touch', 'css', 'database', 'debugging', 'delphi', 'design','design-patterns',\
'django', 'eclipse', 'excel', 'flash', 'flex','html', 'iis', 'internet-explorer', 'iphone', 'java', 'javascript',\
'jquery', 'language-agnostic', 'linq', 'linq-to-sql', 'linux', 'macos','multithreading', 'mysql','nhibernate',\
'objective-c', 'oop', 'oracle','performance', 'perl', 'php', 'python', 'regex', 'ruby','ruby-on-rails', 'security',\
'sharepoint', 'silverlight', 'sql','sql-server', 'sql-server-2005', 'string', 'svn', 'tsql', 'unit-testing',\
'user-interface', 'vb.net', 'version-control','visual-studio', 'visual-studio-2008', 'wcf', 'web-services', 'winapi',\
'windows', 'winforms', 'wpf', 'xml']

# Words that appear most of the time. We will remove them.

most_freq_w=['using','like', 'im', 'would', 'use', 'code', 'get', 'way', 'new', 'want']

# Words that rarely appear. We will remove them.
less_feq_w=['efdestroyworksheetws', 'jltmatrix0length', 'offltagtltligtthis','postmessagemyprojectentitiesbetmessage',\
'columnsfooter', 'omittednote', 'classestaskm95720090413', 'httpossoetikerchrrdtoo', 'addcontrolstolistccontrols',\
'dofollowuplots']

# List of eligible tags for Cluster 0. We got that list for LDA model.
cluster_0=['c#', '.net', 'java', 'asp.net', 'c++', 'javascript', 'python', 'php', 'windows', 'html', 'xml', 'ruby',\
'css', 'linux', 'svn', 'winforms', 'language-agnostic', 'unit-testing', 'vb.net', 'user-interface', 'macos', 'performance',\
'algorithm', 'sql', 'ruby-on-rails', 'version-control', 'web-services', 'database', 'oop', 'multithreading', 'wpf',\
'security', 'visual-studio', 'regex', 'flex', 'asp.net-mvc', 'perl', 'winapi', 'ajax', 'sharepoint', 'flash', 'debugging',\
'jquery', 'eclipse', 'string', 'sql-server', 'apache', 'exception', 'oracle', 'design', 'iis', 'testing', 'actionscript-3',\
'mysql', 'design-patterns', 'delphi', 'wcf', 'arrays', '.net-3.5', 'excel', 'generics', 'browser', 'internet-explorer',\
'silverlight', 'linq', 'memory', 'parsing', 'validation', 'reflection', '.net-2.0', 'visual-studio-2008', 'deployment',\
'optimization', 'networking', 'http', 'web-applications', 'objective-c', 'unix', 'firefox', 'image', 'ide', 'iphone',\
'caching', 'cocoa', 'authentication', 'xslt', 'events', 'collections', 'dom', 'sockets', 'cross-platform', 'date',\
'compiler-construction', 'email', 'serialization', 'moss', 'file', 'tdd', 'django', 'asp.net-ajax']

# List of eligible tags for Cluster 1. We got that list for LDA model.
cluster_1=['sql-server', 'sql', 'database', 'mysql', 'sql-server-2005', '.net', 'oracle', 'tsql', 'c#', 'linq-to-sql',\
'asp.net', 'linq', 'stored-procedures', 'php', 'performance', 'database-design', 'ms-access', 'indexing', 'security',\
'reporting-services', 'vb.net', 'optimization', 'java', 'sql-server-2000', 'postgresql', 'full-text-search', 'sqlite',\
'ruby-on-rails', 'sql-server-2008', 'orm', 'triggers', 'ado.net', 'scripting', 'backup', 'c++', 'ssis', 'xml',\
'visual-studio', 'design-patterns', 'windows', 'replication', 'nhibernate', 'date', 'schema', 'version-control',\
'design', 'transactions', 'unit-testing', '.net-3.5', 'ruby', 'migration', 'entity-framework', 'deployment', 'python',\
'logging', 'datetime', 'web-applications', 'temp-tables', 'macos', 'sql-server-ce', 'architecture', 'search', 'html',\
'join', 'encryption', 'web-services', 'plsql', 'dynamic-data', 'metadata', 'perl', 'installation', 'permissions', 'db2',\
'regex', 'ssms', 'csv', 'sql-injection', 'sharepoint', 'sql-server-express', 'odbc', 'asp-classic', 'css', 'datediff',\
'excel', 'rdbms', 'linux', 'testing', 'dataset', 'statistics', 'email', 'data-structures', 'refactoring', 'iis', 'svn',\
'datatable', 'language-agnostic', 'multithreading', 'hibernate', 'import', 'data-migration']

# List of eligible tags for Cluster 2. We got that list for LDA model.
cluster_2=['visual-studio', 'visual-studio-2008', '.net', 'debugging', 'c++', 'c#', 'version-control', 'asp.net',\
'visual-studio-2005', 'asp.net-mvc', 'iis', 'code-review', 'stl', 'webproject', 'windbg', 'symbols', 'vb.net', 'ide',\
'windows', 'visual-c++', 'svn', 'winforms', 'tfs', 'visual-sourcesafe', 'installation', 'unit-testing', 'intellisense',\
'user-interface', 'msbuild', 'macros', 'projects-and-solutions', 'build', 'resharper', 'multiple-monitors', 'crash',\
'asp-classic', 'wpf', 'vsx', 'extensibility', 'deployment', 'visual-studio-express', 'nant', 'web-services', 'keyboard',\
'continuous-integration', 'eclipse', 'add-in', 'build-automation', 'cruisecontrol.net', 'visual-studio-2008-sp1',\
'visual-c++-2005', 'dependencies', 'gui-designer', 'sql', 'registry', 'python', 'profiling', 'web-applications',\
'javascript', 'editor', 'assemblies', 'silverlight', 'command-line', 'formatting', 'projects', 'vb6', 'vc6',\
'windows-installer', 'shortcuts', 'linux', 'visualsvn-server', 'plugins', 'linq', 'add-on', 'breakpoints',\
'build-process', 'text-editor', 'debuggervisualizer', 'solution', 'windows-xp', 'keyboard-shortcuts', 'reflection',\
'cvs', 'atl', 'mstest', 'testing', 'linq-to-sql', 'open-source', 'development-environment', 'integration', 'oracle',\
'code-generation', 'xml', 'windows-vista', 'xaml', 'wsdl', 'security', 'google-chrome', 'linker', 'code-analysis']

stop=stopwords.words("english")

#Cleaning text and keep key words
def clean_p(text):
    text = " ".join(x.lower() for x in text.split() ) 
    text=re.sub('[^\w\s]', "", text)                           
    stop = stopwords.words('english')
    text = " ".join(x for x in text.split() if x not in stop)
    text = " ".join(x for x in text.split() if x not in most_freq_w)
    text = " ".join(x for x in text.split() if x not in less_feq_w)
    return(text)

#Predicting questions tags with supervised models
def predict_S(text):
    linearSVC = joblib.load('linearSVC_.pkl')
    text=[clean_p(text)]
    predict=linearSVC.predict(text)
    tags=[]
    for i in range(0,len(tag_s)):
        if predict.tolist()[0][i]==1:
            tags.append(tag_s[i])
    tags_="".join('<'+x+'>' for x in tags)
    return tags_

#Predicting questions tags with unsupervised models
def predict_U(text):
    text=text.lower()
    text = " ".join(x for x in text.split() if x not in most_freq_w)
    text = " ".join(x for x in text.split() if x not in less_feq_w)
    text_=[clean_p(text)]
    pkl_file = open('logisticUS.pkl', 'rb')
    logisticUS= pickle.load(pkl_file)
    predict=logisticUS.predict(text_)[0]
    if predict==0:
        predict_tag="".join('<'+x+'>' for x in cluster_0 if x in text.split())
        if predict_tag=='': predict_tag="".join('<'+x+'>' for x in cluster_0[0:5])
    elif predict==1:
        predict_tag="".join('<'+x+'>' for x in cluster_1 if x in text.split())
        if predict_tag=='': predict_tag="".join('<'+x+'>' for x in cluster_1[0:5])
    elif predict==2:
        predict_tag="".join('<'+x+'>' for x in cluster_2 if x in text.split())
        if predict_tag=='': predict_tag="".join('<'+x+'>' for x in cluster_2[0:5])
    return predict_tag
