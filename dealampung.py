import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
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
    st.title("Simulasi Realokasi Anggaran dengan Linear Programming")
    menu = ["Simulasi_LP","Analisis"]
    choice = st.sidebar.selectbox("Select Menu", menu)

    if choice == "Simulasi_LP":
        df = pd.read_excel('Anggaran.xlsx')
        provinsi = st.sidebar.selectbox('Pilih Provinsi',df.Provinsi.unique())
        df = df[df['Provinsi'].isin([provinsi])]
        pemda = st.sidebar.selectbox('Pilih Pemda',df.Pemda.unique())
        # pilihsektor = st.sidebar.selectbox('Pilih Potensi Sektoral',df.Potensi.unique().tolist())
        # df = df[df['Potensi'].isin([pilihsektor])]
        tahun = st.sidebar.selectbox('Pilih Tahun',[2019,2020])
        col = str("y"+str(tahun))
        if tahun==2019:
            dff = pd.read_excel('data2019.xlsx')
            dfc = dff[dff['Pemda'].isin([pemda])]
            st.sidebar.write('Kondisi : Normal')
            

        elif tahun==2020:
            dff = pd.read_excel('data2020.xlsx')
            dfc = dff[dff['Pemda'].isin([pemda])]
            dampak = dfc.Dampak.tolist()
            st.sidebar.write('Kondisi : Pandemi')
            st.sidebar.write(f'Dampak Pandemi : {dampak[0]}')
            # sektor= 'Cluster'
        sektor = 'Sektor_group'
        potensi = dfc.Potensi.tolist()
        st.sidebar.write(f'Sektor_Potensial: {potensi[0]}')
        sektor = dfc[sektor].tolist()
        sektor = sektor[0]
        # st.write(sektor)
        st.sidebar.write(f'Tahun Anggaran  : {tahun}')
        df = df[df['Pemda'].isin([pemda])]
        anggaran = df[col].tolist()
        # st.sidebar.write(f'Total Anggaran Berjalan : {anggaran[0]}')
        st.sidebar.number_input(label="Total Realisasi Anggaran (Milyar Rupiah)",value=int(anggaran[0])/1000000000,min_value=0.0, max_value=1000000000.0, step=10.0)
        nextangg = df[str("y"+str(tahun+1))].tolist()
        # st.sidebar.write(f'Anggaran Tahun Berikutnya : {nextangg[0]}')
        st.sidebar.number_input(label="Total Anggaran Tahun Berikutnya (Milyar Rupiah)",value=int(nextangg[0])/1000000000,min_value=0.0, max_value=1000000000.0, step=10.0)
        efbase = int(dfc.Efisiensi.sum()*10000)/100
        # st.sidebar.write(f'Potensi Sektoral  : {sektor}')
        st.subheader(f'Tingkat Efisiensi Pemda: {efbase} %')
        st.subheader(f'Tingkat Growth PDRB Pemda: {int(dfc.GrowthY.sum()*10000)/100} %')
        s1='PelayananUmum'
        s2='Pendidikan'
        s3='PerlindunganSosial'
        s4='KetertibandanKeamanan'
        s5='Ekonomi'
        s6='LingkunganHidup'
        s7='PerumahandanFasilitasUmum'
        s8='Kesehatan'
        s9='PariwisatadanBudaya'
        dfm = pd.melt(dfc,id_vars=['Pemda'],value_vars=[s1,s2,s3,s4,s5,s6,s7,s8,s9])
        # dflp = dff[dff['Provinsi'].isin([provinsi])]
        dffs = dff[dff['Potensi']==potensi[0]]
        # st.write(provinsi)
        if tahun==2019:
            dflp=dffs
            clno=0
        #     # dflp = dff[dff['Sektor_group']==sektor]
        elif tahun==2020:
            dflp = dffs[dffs['Flag_anomali']==0]
            klasterls = dflp['Dampak'].unique()
            klaster = st.multiselect('Sesuaikan Dampak Pandemi',klasterls,default=dampak[0])
            dflp = dflp[dflp['Dampak'].isin(klaster)]
            clno = dflp.Cluster.tolist()
            clno = clno[0]
            # st.write(clno)
        #     # dffs = dff[dff['Sektor_group']==sektor]
        #     klaster = dflp['Cluster'].tolist()
        #     klaster = klaster[0]
        #     dflp = dflp[dflp['Cluster']==klaster]
        
        dflp = dflp.replace(to_replace=0,value=np.NAN)
        top = dflp['Efisiensi'].max()
        # st.write(top)
        dflp = dflp[dflp['Efisiensi']>=top-0.05]
        dflp['Growth (%)']=dflp['GrowthY']*100
        # dflp = dflp[dflp['Efisiensi'].isin([1])]
        with st.beta_expander('Daftar Pemda Frontier Sektor/Klaster', expanded=False):
            st.write(dflp[['Provinsi','Pemda','Efisiensi','Potensi','Growth (%)']])
        # dflp = dflp.replace(to_replace=0,value=np.NAN)
        kolom = dfm.variable.unique().tolist()
        fig = go.Figure()
        fig.add_trace(go.Box(y=dflp[s1],name=s1))
        fig.add_trace(go.Box(y=dflp[s2],name=s2))
        fig.add_trace(go.Box(y=dflp[s3],name=s3))
        fig.add_trace(go.Box(y=dflp[s4],name=s4))
        fig.add_trace(go.Box(y=dflp[s5],name=s5))
        fig.add_trace(go.Box(y=dflp[s6],name=s6))
        fig.add_trace(go.Box(y=dflp[s7],name=s7))
        fig.add_trace(go.Box(y=dflp[s8],name=s8))
        fig.add_trace(go.Box(y=dflp[s9],name=s9))
        fig.add_trace(go.Scatter(x=kolom, y=dfm['value'],mode='lines',name=pemda))
        fig.update_layout(width=900,height=600,title="Perbandingan Pemda Terpilih Dengan Frontier Sektor/Klaster")
        st.plotly_chart(fig)

        with st.beta_expander('Efficiency Analysis', expanded=False):
            c1,c2,c3,c4 = st.beta_columns((2,1,2,2))
            with c1:
                bv = dfc.iloc[0,1:15].astype('float').tolist()
                st.text_input(label=s1+" (%)",value=int(bv[4]*10000)/100)
                st.text_input(label=s2+" (%)",value=int(bv[5]*10000)/100)
                st.text_input(label=s3+" (%)",value=int(bv[6]*10000)/100)
                st.text_input(label=s4+" (%)",value=int(bv[7]*10000)/100)
                st.text_input(label=s5+" (%)",value=int(bv[8]*10000)/100)
                st.text_input(label=s6+" (%)",value=int(bv[9]*10000)/100)
                st.text_input(label=s7+" (%)",value=int(bv[10]*10000)/100)
                st.text_input(label=s8+" (%)",value=int(bv[11]*10000)/100)
                st.text_input(label=s9+" (%)",value=int(bv[12]*10000)/100)
            with c2:
                st.empty()
            with c3:
                #min value
                v1min = st.number_input(label="PelayananUmum min(%)",value=dflp['PelayananUmum'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v2min = st.number_input(label="Pendidikan min(%)",value=dflp['Pendidikan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v3min = st.number_input(label="PerlindunganSosial min(%)",value=dflp['PerlindunganSosial'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v4min = st.number_input(label="KetertibandanKeamanan min(%)",value=dflp['KetertibandanKeamanan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v5min = st.number_input(label="Ekonomi min(%)",value=dflp['Ekonomi'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v6min = st.number_input(label="LingkunganHidup min(%)",value=dflp['LingkunganHidup'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v7min = st.number_input(label="PerumahandanFasilitasUmum min(%)",value=dflp['PerumahandanFasilitasUmum'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v8min = st.number_input(label="Kesehatan min(%)",value=dflp['Kesehatan'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v9min = st.number_input(label="PariwisatadanBudaya min(%)",value=dflp['PariwisatadanBudaya'].min()*100.0,min_value=0.0, max_value=100.0, step=1.0)
            with c4:
                #max value
                v1max = st.number_input(label="PelayananUmum max (%)",value=dflp['PelayananUmum'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v2max = st.number_input(label="Pendidikan max (%)",value=dflp['Pendidikan'].max()*100,min_value=0.0, max_value=100.0, step=1.0)
                v3max = st.number_input(label="PerlindunganSosial max (%)",value=dflp['PerlindunganSosial'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v4max = st.number_input(label="KetertibandanKeamanan max (%)",value=dflp['KetertibandanKeamanan'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v5max = st.number_input(label="Ekonomi max (%)",value=dflp['Ekonomi'].max()*100.0, max_value=100.0, step=1.0)
                v6max = st.number_input(label="LingkunganHidup max (%)",value=dflp['LingkunganHidup'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v7max = st.number_input(label="PerumahandanFasilitasUmum max (%)",value=dflp['PerumahandanFasilitasUmum'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v8max = st.number_input(label="Kesehatan max (%)",value=dflp['Kesehatan'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)
                v9max = st.number_input(label="PariwisatadanBudaya max (%)",value=dflp['PariwisatadanBudaya'].max()*100.0,min_value=0.0, max_value=100.0, step=1.0)

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
        v1 = LpVariable(name="PelayananUmum", lowBound=0)
        v2 = LpVariable(name="Pendidikan", lowBound=0)
        v3 = LpVariable(name="PerlindunganSosial", lowBound=0)
        v4 = LpVariable(name="KetertibandanKeamanan", lowBound=0)
        v5 = LpVariable(name="Ekonomi", lowBound=0)
        v6 = LpVariable(name="LingkunganHidup", lowBound=0)
        v7 = LpVariable(name="PerumahandanFasilitasUmum", lowBound=0)
        v8 = LpVariable(name="Kesehatan", lowBound=0)
        v9 = LpVariable(name="PariwisatadanBudaya", lowBound=0)
        # , cat="Float")
        bg = dfc.GrowthY.tolist()
        bg = bg[0]
        efscore= ef[0]+ef[1]*bv[0]+ef[2]*bv[1]+ef[3]*bv[2]+ef[4]*bv[3]+v1*ef[5]+v2*ef[6]+v3*ef[7]+v4*ef[8]+v5*ef[9]+v6*ef[10]+v7*ef[11]+v8*ef[12]+v9*ef[13]+sektor*ef[14]+clno*ef[15]
        #Objective
        # prob += ef[0]+ef[1]*bv[1]+ef[2]*bv[2]+ef[3]*bv[4]+ef[4]*bv[5]+v1*ef[5]+v2*ef[6]+v3*ef[7]+v4*ef[8]+v5*ef[9]+v6*ef[10]+v7*ef[11]+v8*ef[12]+v9*ef[13]
        grscore = gr[0]+bv[1]*gr[1]+bv[2]*gr[2]+bv[3]*gr[3]+v1*gr[4]+v2*gr[5]+v3*gr[6]+v4*gr[7]+v5*gr[8]+v6*gr[9]+v7*gr[10]+v8*gr[11]+v9*gr[12]+sektor*gr[13]+clno*gr[14]
        prob += grscore
        prob += efscore
        # prob += grscore
        # Add the constraints to the model
        prob += (v1+v2+v3+v4+v5+v6+v7+v8+v9 <=1, "full_constraint")
        prob += (v1*100 >= v1min, "v1min")
        prob += (v2*100 >= v2min, "v2min")
        prob += (v3*100 >= v3min, "v3min")
        prob += (v4*100 >= v4min, "v4min")
        prob += (v5*100 >= v5min, "v5min")
        prob += (v6*100 >= v6min, "v6min")
        prob += (v7*100 >= v7min, "v7min")
        prob += (v8*100 >= v8min, "v8min")
        prob += (v9*100 >= v9min, "v9min")
        prob += (v1*100 <= v1max, "v1max")
        prob += (v2*100 <= v2max, "v2max")
        prob += (v3*100 <= v3max, "v3max")
        prob += (v4*100 <= v4max, "v4max")
        prob += (v5*100 <= v5max, "v5max")
        prob += (v6*100 <= v6max, "v6max")
        prob += (v7*100 <= v7max, "v7max")
        prob += (v8*100 <= v8max, "v8max")
        prob += (v9*100 <= v9max, "v9max")
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
            p7 =  pulp.value(v7)
            p8 =  pulp.value(v8)
            p9 =  pulp.value(v9)
            total = int((p1+p2+p3+p4+p5+p6+p7+p8+p9)*10000)/100
            outls = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
            # st.subheader(outls)
            h1,h2 = st.beta_columns((5,3))
            
            with h1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(x=kolom, y=dfm['value'],name='Current Allocation'))
                fig1.add_trace(go.Bar(x=kolom, y=outls,name='Recommendation'))
                fig1.update_layout(width=700, height=600)
                st.plotly_chart(fig1)
            with h2:
                # efficiency= ef[0]+ef[1]*bv[1]+ef[2]*bv[2]+ef[3]*bv[4]+ef[4]*bv[5]+p1*ef[5]+p2*ef[6]+p3*ef[7]+p4*ef[8]+p5*ef[9]+p6*ef[10]+p7*ef[11]+p8*ef[12]+p9*ef[13]
                growth= gr[0]+bv[1]*gr[1]+bv[2]*gr[2]+bv[3]*gr[3]+p1*gr[4]+p2*gr[5]+p3*gr[6]+p4*gr[7]+p5*gr[8]+p6*gr[9]+p7*gr[10]+p8*gr[11]+p9*gr[12]+sektor*gr[13]+clno*gr[14]
                efficiency= ef[0]+ef[1]*bv[0]+ef[2]*bv[1]+ef[3]*bv[2]+ef[4]*bv[3]+p1*ef[5]+p2*ef[6]+p3*ef[7]+p4*ef[8]+p5*ef[9]+p6*ef[10]+p7*ef[11]+p8*ef[12]+p9*ef[13]+sektor*ef[14]+clno*ef[15]
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
                                value = int(growth*100)/100,
                                title = {"text": "Prediksi Tingkat Growth (%):"},
                                delta = {'reference': int(dfc.GrowthY.sum()*10000)/100, 'relative': False},
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
                # fig3.update_layout(width=200)
                st.plotly_chart(fig3)
            st.subheader(f'Tingkat Alokasi Anggaran: {int(total)}%')
            with st.beta_expander("Selisih Lebih/Kurang Anggaran dari Realokasi",expanded=False):
                excess = int(anggaran[0]) * (100-total) /100000000000
                st.number_input(label=" ",value=excess,min_value=0.0, max_value=1000000000.0, step=10.0)
            with st.beta_expander("Prediksi Kenaikan (Penurunan) PDRB",expanded=False):
                gap = int(anggaran[0]) * int(growth-int(dfc.GrowthY.sum()*10000)) /10000000000000
                # if st.button("Klik untuk Jalankan"):
                # st.sidebar.number_input(label="Nilai PDRB Kondisi Saat ini (Milyar Rupiah)",value=int(anggaran[0])/1000000000,min_value=0.0, max_value=1000000000.0, step=10.0)
                # st.sidebar.number_input(label="Nilai PDRB dengan Alokasi Baru (Milyar Rupiah)",value=int(anggaran[0])/1000000000,min_value=0.0, max_value=1000000000.0, step=10.0)
                st.number_input(label="Selisih Lebih/Kurang PDRB dari Realokasi (Milyar Rupiah)",value=gap,min_value=0-anggaran[0]/1000000000, max_value=1000000000.0, step=10.0)
                st.number_input(label="Prediksi Kenaikan/Penurunan PDRB Berdasarkan Anggaran Tahun Berikutnya (Milyar Rupiah)",value=growth*int(nextangg[0])/100000000000,min_value=0-nextangg[0]/1000000000, max_value=1000000000.0, step=10.0)
                # st.empty()
            
if __name__=='__main__':
    main()