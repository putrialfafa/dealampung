import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import math
# from PIL import Image
st.set_page_config(layout='wide')
# import plotly.io as pio
# pio.templates
# pio.templates.default = "simple_white"
# load model 
# import joblib

# linear programming
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

def main():
    st.title("Simulasi Realokasi Anggaran Dana Desa dengan Linear Programming")
    menu = ["Simulasi_LP","Analisis"]
    choice = st.sidebar.selectbox("Select Menu", menu)

    if choice == "Simulasi_LP":
        df = pd.read_excel('Anggaran.xlsx')
        Kabupaten = st.sidebar.selectbox('Pilih Kabupaten',df.Kabupaten.unique())
        df = df[df['Kabupaten'].isin([Kabupaten])]
        Kecamatan = st.sidebar.selectbox('Pilih Kecamatan',df.Kecamatan.unique())
        df = df[df['Kecamatan'].isin([Kecamatan])]
        Desa = st.sidebar.selectbox('Pilih Desa',df.Desa.unique())
        # pilihsektor = st.sidebar.selectbox('Pilih Potensi Sektoral',df.Potensi.unique().tolist())
        # df = df[df['Potensi'].isin([pilihsektor])]
        tahun = st.sidebar.selectbox('Pilih Tahun',[2020,2021])
        col = str("y"+str(tahun))
        if tahun==2020:
            dff = pd.read_excel('data2020.xlsx')
            dfc = dff[dff['Desa'].isin([Desa])]
            # st.sidebar.write('Kondisi : Normal')
            

        elif tahun==2021:
            dff = pd.read_excel('data2021.xlsx')
            dfc = dff[dff['Desa'].isin([Desa])]
            # dampak = dfc.Dampak.tolist()
            # st.sidebar.write('Kondisi : Pandemi')
            # st.sidebar.write(f'Dampak Pandemi : {dampak[0]}')
            # sektor= 'Cluster'
        sektor = 'Sektor_group'
        #Status_IDM = dfc.Status_IDM.tolist()
        sektor = dfc[sektor].tolist()
        sektor = sektor[0]
        #IDM = 'Status_IDM'
        Status_IDM = dfc.Status_IDM.tolist()
        st.sidebar.write(f'Status IDM: {Status_IDM[0]}')
        #IDM = dfc[IDM].tolist()
        #IDM = IDM[0]
        # st.write(sektor)
        st.sidebar.subheader('Nilai Anggaran Tahun Berjalan:')
        st.sidebar.write(f'Tahun Anggaran: {tahun}')
        df = df[df['Desa'].isin([Desa])]
        anggaran = df[col].tolist()
        st.sidebar.write("Rp {:,.0f}".format(anggaran[0]))
        adj = st.sidebar.number_input(label="Penyesuaian Anggaran (%)",value=0.0,min_value=0.0, max_value=100.0, step=5.0)
        p_anggaran = [anggaran[0] +  (anggaran[0]*adj/100),0]
        st.sidebar.subheader('Alokasi Anggaran Setelah Penyesuaian:')
        st.sidebar.write("Rp {:,.0f}".format(p_anggaran[0]))
        st.sidebar.subheader('Rencana Anggaran Tahun Berikutnya:')
        nextangg = df[str("y"+str(tahun+1))].tolist()
        st.sidebar.write("Rp {:,.0f}".format(nextangg[0]))
        # st.sidebar.write(f'Total Anggaran Berjalan : {anggaran[0]}')
        # st.sidebar.number_input(label="Total Anggaran Berjalan",value=int(anggaran[0]),min_value=0, max_value=1000000000000, step=100000000)
        # nextangg = df[str("y"+str(tahun+1))].tolist()
        # st.sidebar.write(f'Anggaran Tahun Berikutnya : {nextangg[0]}')
        # st.sidebar.number_input(label="Total Anggaran Tahun Berikutnya",value=int(nextangg[0]),min_value=0, max_value=1000000000000, step=100000000)
        efbase = int(dfc.Efisiensi.sum()*10000)/100
        # st.sidebar.write(f'Potensi Sektoral  : {sektor}')
        st.subheader(f'Tingkat Efisiensi Desa: {efbase} %')
        st.subheader(f'Indeks Desa Membangun: {int(dfc.IDM.sum()*100)/100}')
        s1='PelaksanaanPembangunanDesa'
        s2='PemberdayaanMasyarakatDesa'
        s3='Pembiayaan'
        s4='PembinaanKemasyarakatanDesa'
        s5='PenanggulanganBencanaKeadaanDaruratDanMendesakDesa'
        s6='PenyelenggaraanPemerintahanDesa'
        dfm = pd.melt(dfc,id_vars=['Desa'],value_vars=[s1,s2,s3,s4,s5,s6])
        # dflp = dff[dff['Provinsi'].isin([provinsi])]
        dffs = dff[dff['Status_IDM']==Status_IDM[0]]
        # st.write(provinsi)
        if tahun==2020:
            dflp=dffs
            # clno=0
        #     # dflp = dff[dff['Sektor_group']==sektor]
        elif tahun==2021:
            dflp = dffs[dffs['Flag_anomali']==0]
            # klasterls = dflp['Dampak'].unique()
            # klaster = st.multiselect('Sesuaikan Dampak Pandemi',klasterls,default=dampak[0])
            # dflp = dflp[dflp['Dampak'].isin(klaster)]
            # clno = 0
            # st.write(clno)
        #     # dffs = dff[dff['Sektor_group']==sektor]
        #     klaster = dflp['Cluster'].tolist()
        #     klaster = klaster[0]
        #     dflp = dflp[dflp['Cluster']==klaster]
        
        dflp = dflp.replace(to_replace=0,value=np.NAN)
        top = dflp['Efisiensi'].max()
        # st.write(top)
        dflp = dflp[dflp['Efisiensi']>=top-0.05]
        dflp['IDM']=dflp['IDM']
        # dflp = dflp[dflp['Efisiensi'].isin([1])]
        with st.expander('Daftar Desa Frontier', expanded=False):
            st.write(dflp[['Kabupaten','Kecamatan','Desa','Efisiensi','IDM','Status_IDM']])
        # dflp = dflp.replace(to_replace=0,value=np.NAN)
        kolom = dfm.variable.unique().tolist()
        fig = go.Figure()
        fig.add_trace(go.Box(y=dflp[s1],name=s1))
        fig.add_trace(go.Box(y=dflp[s2],name=s2))
        fig.add_trace(go.Box(y=dflp[s3],name=s3))
        fig.add_trace(go.Box(y=dflp[s4],name=s4))
        fig.add_trace(go.Box(y=dflp[s5],name=s5))
        fig.add_trace(go.Box(y=dflp[s6],name=s6))
        #fig.add_trace(go.Box(y=dflp[s7],name=s7))
        #fig.add_trace(go.Box(y=dflp[s8],name=s8))
        #fig.add_trace(go.Box(y=dflp[s9],name=s9))
        fig.add_trace(go.Scatter(x=kolom, y=dfm['value'],mode='lines',name=Desa))
        fig.update_layout(width=900,height=600,title="Perbandingan Desa Terpilih Dengan Frontier")
        st.plotly_chart(fig)

        with st.expander('Efficiency Analysis', expanded=False):
            c1,c2,c3,c4 = st.columns((5,1,5,5))
            with c1:
                bv = dfc.iloc[0,1:9].astype('int').tolist()
                st.text_input(label=s1,value=int(bv[1]))
                st.text_input(label=s2,value=int(bv[2]))
                st.text_input(label=s3,value=int(bv[3]))
                st.text_input(label=s4,value=int(bv[4]))
                st.text_input(label=s5,value=int(bv[5]))
                st.text_input(label=s6,value=int(bv[6]))
                #st.text_input(label=s7+" (%)",value=int(bv[10]*10000)/100)
                #st.text_input(label=s8+" (%)",value=int(bv[11]*10000)/100)
                #st.text_input(label=s9+" (%)",value=int(bv[12]*10000)/100)
            with c2:
                st.empty()
            with c3:
                #min value
                v1min = st.number_input(label="PelaksanaanPembangunanDesa min",value=dflp['PelaksanaanPembangunanDesa'].fillna(0).astype(int).min(),min_value=0, max_value=1000000000000, step=1000000)
                v2min = st.number_input(label="PemberdayaanMasyarakatDesa min",value=dflp['PemberdayaanMasyarakatDesa'].fillna(0).astype(int).min(),min_value=0, max_value=1000000000000, step=1000000)
                v3min = st.number_input(label="Pembiayaan min",value=dflp['Pembiayaan'].fillna(0).astype(int).min(),min_value=0, max_value=1000000000000, step=1000000)
                v4min = st.number_input(label="PembinaanKemasyarakatanDesa min",value=dflp['PembinaanKemasyarakatanDesa'].fillna(0).astype(int).min(),min_value=0, max_value=1000000000000, step=1000000)
                v5min = st.number_input(label="PenanggulanganBencanaKeadaanDaruratDanMendesakDesa min",value=dflp['PenanggulanganBencanaKeadaanDaruratDanMendesakDesa'].fillna(0).astype(int).min(),min_value=0, max_value=1000000000000, step=1000000)
                v6min = st.number_input(label="PenyelenggaraanPemerintahanDesa min",value=dflp['PenyelenggaraanPemerintahanDesa'].fillna(0).astype(int).min(),min_value=0, max_value=1000000000000, step=1000000)
                #v1min = st.number_input(label="PelaksanaanPembangunanDesa min(%)",value=dflp['X1'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v2min = st.number_input(label="PemberdayaanMasyarakatDesa min(%)",value=dflp['X2'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v3min = st.number_input(label="Pembiayaan min(%)",value=dflp['X3'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v4min = st.number_input(label="PembinaanKemasyarakatanDesa min(%)",value=dflp['X4'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v5min = st.number_input(label="PenanggulanganBencanaKeadaanDaruratDanMendesakDesa min(%)",value=dflp['X5'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v6min = st.number_input(label="PenyelenggaraanPemerintahanDesa min(%)",value=dflp['X6'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v7min = st.number_input(label="PerumahandanFasilitasUmum min(%)",value=dflp['PerumahandanFasilitasUmum'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v8min = st.number_input(label="Kesehatan min(%)",value=dflp['Kesehatan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v9min = st.number_input(label="PariwisatadanBudaya min(%)",value=dflp['PariwisatadanBudaya'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v1minn = [v1min * p_anggaran[0],0]
                #v2minn = [v2min * p_anggaran[0],0]
                #v3minn = [v3min * p_anggaran[0],0]
                #v4minn = [v4min * p_anggaran[0],0]
                #v5minn = [v5min * p_anggaran[0],0]
                #v6minn = [v6min * p_anggaran[0],0]
            with c4:
                #max value
                v1max = st.number_input(label="PelaksanaanPembangunanDesa max",value=dflp['PelaksanaanPembangunanDesa'].fillna(0).astype(int).max(),min_value=0, max_value=1000000000000, step=1000000)
                v2max = st.number_input(label="PemberdayaanMasyarakatDesa max",value=dflp['PemberdayaanMasyarakatDesa'].fillna(0).astype(int).max(),min_value=0, max_value=1000000000000, step=1000000)
                v3max = st.number_input(label="Pembiayaan max",value=dflp['Pembiayaan'].fillna(0).astype(int).max(),min_value=0, max_value=1000000000000, step=1000000)
                v4max = st.number_input(label="PembinaanKemasyarakatanDesa max",value=dflp['PembinaanKemasyarakatanDesa'].fillna(0).astype(int).max(),min_value=0, max_value=1000000000000, step=1000000)
                v5max = st.number_input(label="PenanggulanganBencanaKeadaanDaruratDanMendesakDesa max",value=dflp['PenanggulanganBencanaKeadaanDaruratDanMendesakDesa'].fillna(0).astype(int).max(),min_value=0, max_value=1000000000000, step=1000000)
                v6max = st.number_input(label="PenyelenggaraanPemerintahanDesa max",value=dflp['PenyelenggaraanPemerintahanDesa'].fillna(0).astype(int).max(),min_value=0, max_value=1000000000000, step=1000000)
                #v1max = st.number_input(label="PelaksanaanPembangunanDesa max (%)",value=dflp['X1'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v2max = st.number_input(label="PemberdayaanMasyarakatDesa max (%)",value=dflp['X2'].max()*100,min_value=0.0, max_value=100.0, step=1.0)
                #v3max = st.number_input(label="Pembiayaan max (%)",value=dflp['X3'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v4max = st.number_input(label="PembinaanKemasyarakatanDesa max (%)",value=dflp['X4'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v5max = st.number_input(label="PenanggulanganBencanaKeadaanDaruratDanMendesakDesa max (%)",value=dflp['X5'].max()*100.0, max_value=100.0, step=1.0)
                #v6max = st.number_input(label="PenyelenggaraanPemerintahanDesa max (%)",value=dflp['X6'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v7max = st.number_input(label="PerumahandanFasilitasUmum max (%)",value=dflp['PerumahandanFasilitasUmum'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v8max = st.number_input(label="Kesehatan max (%)",value=dflp['Kesehatan'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v9max = st.number_input(label="PariwisatadanBudaya max (%)",value=dflp['PariwisatadanBudaya'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                #v1maxx = [v1max * p_anggaran[0],0]
                #v2maxx = [v2max * p_anggaran[0],0]
                #v3maxx = [v3max * p_anggaran[0],0]
                #v4maxx = [v4max * p_anggaran[0],0]
                #v5maxx = [v5max * p_anggaran[0],0]
                #v6maxx = [v6max * p_anggaran[0],0]

        # load model
        ef = pd.read_excel('efmodel.xlsx')
        ef = ef[col].tolist()
        gr = pd.read_excel('grwmodel.xlsx')
        gr = gr[col].tolist()
        # ef1 = pd.read_excel('efmodel.xlsx')
        # ef1 = ef1[col].tolist()

        # Create the LP model
        prob = LpProblem(name="Allocation Optimization",sense=LpMaximize)
        # Initialize the decision variables
        v1 = LpVariable(name="PelaksanaanPembangunanDesa", lowBound=0)
        v2 = LpVariable(name="PemberdayaanMasyarakatDesa", lowBound=0)
        v3 = LpVariable(name="Pembiayaan", lowBound=0)
        v4 = LpVariable(name="PembinaanKemasyarakatanDesa", lowBound=0)
        v5 = LpVariable(name="PenanggulanganBencanaKeadaanDaruratDanMendesakDesa", lowBound=0)
        v6 = LpVariable(name="PenyelenggaraanPemerintahanDesa", lowBound=0)
        #v7 = LpVariable(name="PerumahandanFasilitasUmum", lowBound=0)
        #v8 = LpVariable(name="Kesehatan", lowBound=0)
        #v9 = LpVariable(name="PariwisatadanBudaya", lowBound=0)
        # , cat="Float")
        bg = dfc.IDM.tolist()
        bg = bg[0]
        efscore= ef[0]+ef[1]*bv[0]+v1*ef[2]+v2*ef[3]+v3*ef[4]+v4*ef[5]+v5*ef[6]+v6*ef[7]
        # efscore = ef[0]+ef[1]*np.log(bv[0])+ef[2]*np.log(v1)+ef[3]*np.log(v2)+ef[4]*np.log(v3)+ef[5]*np.log(v4)+ef[6]*np.log(v5)+ef[7]*np.log(v6)
        # efscoree = ef[0]+ef[1]*math.log(bv[0])+ef[2]*math.log(v1)+ef[3]*math.log(v2)+math[4]*math.log(v3)+ef[5]*math.log(v4)+ef[6]*math.log(v5)+ef[7]*math.log(v6)
        # efscore = math.exp(efscoree)
        #Objective
        # prob += ef[0]+ef[1]*bv[1]+ef[2]*bv[2]+ef[3]*bv[4]+ef[4]*bv[5]+v1*ef[5]+v2*ef[6]+v3*ef[7]+v4*ef[8]+v5*ef[9]+v6*ef[10]+v7*ef[11]+v8*ef[12]+v9*ef[13]
        grscore = gr[0]+v1*gr[1]+v2*gr[2]+v3*gr[3]+v4*gr[4]+v5*gr[5]+v6*gr[6]
        # grscore = gr[0]+gr[1]**np.log(v1)+gr[2]**np.log(v2)+gr[3]**np.log(v3)+gr[4]**np.log(v4)+gr[5]**np.log(v5)+gr[6]**np.log(v6)
        prob += grscore
        prob += efscore
        # prob += grscore
        # Add the constraints to the model
        prob += (v1+v2+v3+v4+v5+v6 <= int(p_anggaran[0]), "full_constraint")
        prob += (v1 >= v1min, "v1min")
        prob += (v2 >= v2min, "v2min")
        prob += (v3 >= v3min, "v3min")
        prob += (v4 >= v4min, "v4min")
        prob += (v5 >= v5min, "v5min")
        prob += (v6 >= v6min, "v6min")
        #prob += (v7*100 >= v7min, "v7min")
        #prob += (v8*100 >= v8min, "v8min")
        #prob += (v9*100 >= v9min, "v9min")
        prob += (v1 <= v1max, "v1max")
        prob += (v2 <= v2max, "v2max")
        prob += (v3 <= v3max, "v3max")
        prob += (v4 <= v4max, "v4max")
        prob += (v5 <= v5max, "v5max")
        prob += (v6 <= v6max, "v6max")
        #prob += (v7*100 <= v7max, "v7max")
        #prob += (v8*100 <= v8max, "v8max")
        #prob += (v9*100 <= v9max, "v9max")
        # prob += (grscore >= bg, "minGrw")
        # prob += (efscore >=0.5, "minEff")
        prob += (efscore <=1, "maxEff")
        prob += (efscore >=0, "minEff")

        # Solve the problem
        st.write("Penghitungan Alokasi Anggaran Paling Efisien")
        if st.button("Klik untuk Jalankan"):
            status = prob.solve()
            p1 =  pulp.value(v1)
            p2 =  pulp.value(v2)
            p3 =  pulp.value(v3)
            p4 =  pulp.value(v4)
            p5 =  pulp.value(v5)
            p6 =  pulp.value(v6)
            #p7 =  pulp.value(v7)
            #p8 =  pulp.value(v8)
            #p9 =  pulp.value(v9)
            #total = int((p1+p2+p3+p4+p5+p6)*10000)/100
            total = int(p1+p2+p3+p4+p5+p6)
            outls = [p1,p2,p3,p4,p5,p6]
            # st.subheader(outls)
            h1,h2 = st.columns((5,3))
            
            with h1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=kolom, y=dfm['value'],name='Current Allocation'))
                fig1.add_trace(go.Bar(x=kolom, y=outls,name='Recommendation'))
                fig1.update_layout(width=700, height=600)
                st.plotly_chart(fig1)
                newval = [element for element in outls]
                dfm['current_allocation'] = [element for element in dfm['value'].tolist()]
                dfm['after_reallocation'] = newval
                dfm['adjustment'] = dfm['after_reallocation']-dfm['current_allocation']
                dfm = dfm[['variable','current_allocation','adjustment','after_reallocation']]
                dfm = dfm.round({'current_allocation':0, 'adjustment':0, 'after_reallocation':0})
                st.dataframe(dfm.style.format(subset=['current_allocation','adjustment','after_reallocation'], formatter="Rp {:,.0f}"))
            with h2:
                # efficiency= ef[0]+ef[1]*bv[1]+ef[2]*bv[2]+ef[3]*bv[4]+ef[4]*bv[5]+p1*ef[5]+p2*ef[6]+p3*ef[7]+p4*ef[8]+p5*ef[9]+p6*ef[10]+p7*ef[11]+p8*ef[12]+p9*ef[13]
                Indeks = gr[0]+p1*gr[1]+p2*gr[2]+p3*gr[3]+p4*gr[4]+p5*gr[5]+p6*gr[6]
                efficiency= ef[0]+ef[1]*bv[0]+p1*ef[2]+p2*ef[3]+p3*ef[4]+p4*ef[5]+p5*ef[6]+p6*ef[7]
                # growth = gr[0]+bv[1]*gr[1]+bv[2]*gr[2]+bv[3]*gr[3]+bv[4]*gr[4]+bv[5]*gr[5]+bv[6]*gr[6]+bv[7]*gr[7]+bv[8]*gr[8]+bv[9]*gr[9]+bv[10]*gr[10]+bv[11]*gr[11]+bv[12]*gr[12]+bv[13]*gr[13]
                # st.title(f'Prediksi Nilai Growth: {growth*100}%')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                # st.write(status)
                fig3 = go.Figure()
                fig3.add_trace(go.Indicator(
                                mode = "number+delta",
                                # value = status*100,
                                value = int(Indeks*100)/100,
                                title = {"text": "Prediksi Indeks Desa Membangun:"},
                                delta = {'reference': int(dfc.IDM.sum()*10000)/100, 'relative': False},
                                domain = {'x': [0, 0.5], 'y': [0.6, 1]},
                                ))
                fig3.add_trace(go.Indicator(
                                mode = "number+delta",
                                # value = status*100,
                                value = int(efficiency*10000)/100,
                                title = {"text": "Tingkat Efisiensi (%):"},
                                delta = {'reference': int(dfc.Efisiensi.sum()*10000)/100, 'relative': False},
                                domain = {'x': [0, 0.5], 'y': [0, 0.4]},
                                ))
                # fig3.update_layout(width=150)
                st.plotly_chart(fig3)
            st.subheader(f'Tingkat Alokasi Anggaran: {int(total)}%')
            with st.expander("Selisih Lebih/Kurang Anggaran dari Realokasi",expanded=False):
                excess = p_anggaran[0] * (100-total) /100
                # st.number_input(label=" ",value=excess,min_value=0.0, max_value=1000000000.0, step=10.0)
                # st.write("Selisih Lebih/Kurang PDRB dari Realokasi:")
                st.subheader("Rp {:,.0f}".format(excess))
            # with st.expander("Prediksi Kenaikan (Penurunan) IDM dari Realokasi",expanded=False):
                # gap = int(Indeks-int(dfc.IDM.sum()*10000)) /10000
                # gap = int(anggaran[0]) * int(growth-int(dfc.GrowthY.sum()*10000)) /10000000000000
                # if st.button("Klik untuk Jalankan"):
                # st.sidebar.number_input(label="Nilai PDRB Kondisi Saat ini (Milyar Rupiah)",value=int(anggaran[0])/1000000000,min_value=0.0, max_value=1000000000.0, step=10.0)
                # st.sidebar.number_input(label="Nilai PDRB dengan Alokasi Baru (Milyar Rupiah)",value=int(anggaran[0])/1000000000,min_value=0.0, max_value=1000000000.0, step=10.0)
                # st.number_input(label="Selisih Lebih/Kurang PDRB dari Realokasi",value=gap,min_value=0-anggaran[0]/1000000000, max_value=1000000000.0, step=10.0)
                # st.write("Selisih Lebih/Kurang PDRB dari Realokasi:")
                # st.subheader("Rp {:,.0f}".format(gap))
                # st.subheader(f'{gap}')
                # st.write("Prediksi Kenaikan/Penurunan PDRB Berdasarkan Anggaran Tahun Berikutnya:")
                # st.subheader("Rp {:,.0f}".format(growth*int(nextangg[0])))
                # st.number_input(label="Prediksi Kenaikan/Penurunan PDRB Berdasarkan Anggaran Tahun Berikutnya",value=growth*int(nextangg[0])/1000000000,min_value=0-nextangg[0]/1000000000, max_value=1000000000.0, step=10.0)
                # st.empty()
            
if __name__=='__main__':
    main()