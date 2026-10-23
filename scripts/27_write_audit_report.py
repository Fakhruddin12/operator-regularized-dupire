"""Build the evidence report after every benchmark and diagnostic completes."""
from pathlib import Path
import json,sys,html
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
STAGE=ROOT/'outputs/stage_23_literature_audit_final'
EVIDENCE=ROOT/'reports/literature_audit'
TARGET=ROOT/'reports/Dupire_Literature_Audit_Report.pdf'


def main():
    metrics=pd.read_csv(EVIDENCE/'market_and_bates_metrics.csv')
    synth=pd.read_csv(EVIDENCE/'synthetic_recovery_summary.csv')
    pairs=pd.read_csv(EVIDENCE/'paired_comparisons.csv')
    synthpairs=pd.read_csv(EVIDENCE/'synthetic_paired_comparisons.csv')
    numerical=json.loads((EVIDENCE/'numerical_summary.json').read_text())
    audit=json.loads((ROOT/'outputs/implementation_audit.json').read_text())
    environment=json.loads((ROOT/'outputs/audit_environment.json').read_text())
    generation=json.loads((STAGE/'albani_generation_diagnostic.json').read_text())
    math=json.loads((ROOT/'outputs/audit_math_checks.json').read_text())
    input_checks=json.loads((ROOT/'outputs/audit_input_diagnostics.json').read_text())
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('DV',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DV-Bold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyAudit',fontName='DV',fontSize=9.4,leading=13.4,spaceAfter=7,textColor=colors.HexColor('#243246')))
    styles.add(ParagraphStyle(name='SmallAudit',parent=styles['BodyAudit'],fontSize=8,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='TitleAudit',fontName='DV-Bold',fontSize=25,leading=30,spaceAfter=16,textColor=colors.HexColor('#152d45')))
    styles.add(ParagraphStyle(name='HeadingAudit',fontName='DV-Bold',fontSize=15,leading=20,spaceAfter=11,textColor=colors.HexColor('#173e59')))
    styles.add(ParagraphStyle(name='SubAudit',fontName='DV-Bold',fontSize=10.5,leading=14,spaceBefore=6,spaceAfter=7,textColor=colors.HexColor('#173e59')))
    story=[]
    def p(text,small=False):story.append(Paragraph(text,styles['SmallAudit' if small else 'BodyAudit']))
    def h(text):story.append(Paragraph(text,styles['HeadingAudit']))
    def sub(text):story.append(Paragraph(text,styles['SubAudit']))
    def table(rows,widths=None):
        cells=[[Paragraph(html.escape(str(c)),styles['SmallAudit']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8b9eac')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
        story.append(t);story.append(Spacer(1,10))
    def image(name,width=490,height=None):
        from PIL import Image as PILImage
        path=EVIDENCE/name
        w,hp=PILImage.open(path).size
        story.append(Image(str(path),width=width,height=height or width*hp/w));story.append(Spacer(1,7))
    def new():story.append(PageBreak())
    names={'eurostoxx_2010':'Euro Stoxx 50','bates_author':'Bates synthetic'}
    protocols={'quote_cv':'Quote holdout','expiry_cv':'Expiry holdout'}
    methodnames={'ssvi_pde':'SSVI - PDE','h1':'H1 smoothness','h2':'H2 smoothness','operator_full':'Full operator'}
    story.append(Paragraph('Dupire calibration:<br/>implementation audit and<br/>literature benchmarks',styles['TitleAudit']))
    p('Fakhruddin Hakim Hussain | Research evidence report | 23 October 2026')
    sub('What this report establishes')
    p('The intended model is implemented as a full-domain SSVI-Dupire reference multiplied by a compact correction in log local variance. Two defects were fixed: unsafe cubic ATM-variance extrapolation and boundary overcounting in regularization quadrature. The audit adds a verified fixed-grid pricing/sensitivity engine and training-only nested validation.')
    p('<b>The evidence supports a narrower conclusion than a general claim of a superior regularizer.</b> A correction can improve SSVI pricing, while conventional smoothness penalties can match or beat the proposed density-and-wing penalty. All outcomes below are reported; no outer test result selected the method or its tuning parameter.')
    p(f'{numerical["training_only_polished_fits"]} of 250 selected correction fits reached the Gauss-Newton budget; only {numerical["polished_optimizer_success"]} subsequent refinements returned optimizer-success flags. These results compare finite-budget procedures, not proven global minimizers.',True)
    rows=[['Data / held-out object','SSVI','H1','H2','Full operator']]
    for ds in names:
        for proto in protocols:
            d=metrics[(metrics.dataset==ds)&(metrics.protocol==proto)].set_index('method')
            rows.append([names[ds]+' / '+protocols[proto],*[f'{d.loc[m,"price_rmse"]*1e4:.3f}' for m in ['ssvi_pde','h1','h2','operator_full']]])
    table(rows,[177,72,72,72,92])
    p('Price RMSE is measured in basis points of spot (normalized price error multiplied by 10,000). Lower is better. These are refined-grid, out-of-fold prices. Euro Stoxx has 155 historical quotes; Bates has 324 synthetic observations. Each quote is held out exactly once per protocol.',True)
    d=synth[(synth.region=='full')].set_index('method')
    p(f'<b>Known-truth recovery is unstable:</b> across ten noise draws, mean full-region local-volatility RMSE is {d.loc["operator_full","mean"]:.5f} for the full operator versus {d.loc["ssvi_pde","mean"]:.5f} for SSVI. The medians are {d.loc["operator_full","median"]:.5f} and {d.loc["ssvi_pde","median"]:.5f}; the full operator wins only 5/10 draws. A lower quote error does not establish a more accurate recovered surface.',True)
    new();h('1. Mathematical intent and implementation findings')
    p('Let a denote the latent grid amplitude and q the fixed taper. The actual log-variance change is q a, so the final model is:')
    table([['Object','Definition'],['Local variance','u = log(sigma_local^2) = u_SSVI + q a'],['Objective','mean_i [ (PDE_i(a) - observed_i) / scale_i ]^2 + lambda a^T R a / p'],['Operator','R = beta I + alpha_x Dx^T Dx + alpha_T Dt^T Dt + diag(V_density + V_wing)'],['Important distinction','The penalty acts on latent a, not on q a. Natural regularizer boundary behavior, taper boundaries and price-PDE boundaries are different objects.']],[111,374])
    p('The operator expression is shorthand for the intended quadratic energy. Its implemented matrix includes nodal and derivative-edge quadrature weights, followed by trace normalization. The theory note writes those mass matrices explicitly.',True)
    for finding in audit['findings']:
        sub(finding['id']+' | '+finding['severity'])
        p(html.escape(finding['description'])+' <b>Resolution:</b> '+html.escape(finding['fix']),True)
    sub('Checks on the core mathematics')
    p('The Jacobian sign, inverse-variance weighting and penalty on the total latent amplitude, rather than just its update step, agree with the stated nonlinear objective. Volatility is positive through its exponential parameterization. All tested penalty matrices are symmetric positive definite. Exact discrete tangent sensitivities agree with centered finite differences. The original 54 tests and seven new audit tests pass (61 total).',True)
    new();h('2. Inputs and comparison protocol')
    table([['Benchmark','Source and use'],['Euro Stoxx 50','Public QuantLib transcription of the Andreasen-Huge benchmark reused by Geng et al.; 155 quotes, 12 maturities, 1 March 2010. Original quoted maturity labels retained; rates/dividend convention both zero.'],['Bates synthetic','The unchanged 324-row training file from the official Ackerer et al. NeurIPS supplement. Puts converted to normalized calls by parity. Supplied IVs reconstruct prices to machine precision.'],['Albani synthetic','Published piecewise local-volatility truth and 5-by-31 observation layout, with ten new 1% noise draws. Spot is known and fixed at one; the authors spot-adjustment exercise is not replicated.']],[108,377])
    sub('Nested separation')
    p('For each of two supplied datasets: five interleaved strike folds and one fold for each complete expiry (12 Euro Stoxx + 18 Bates). For each synthetic noise realization: a fixed interleaved outer fold. Total: 50 outer fits. Every outer training set gets an inner interleaved 4-way split; inner fold 1 selects lambda. SSVI, quote density and corrections use only the appropriate training subset. The final reference and correction are refitted on outer training observations. Test prices are never used for selection.')
    p('Each penalty gets the same nonlinear procedure, a seven-point lambda grid from 0.0001 to 100, and one two-point extension if the chosen lambda lies at an edge. Selecting the unchanged reference is also allowed. The same known query coordinates define the computational domain for all methods; this is a cross-sectional reconstruction experiment, not prediction at unknown future input locations.')
    sub('Shared model and numerical choices')
    p('The primary correction has 7 maturity by 15 log-moneyness nodes. Relative coefficients are alpha_x=0.005, alpha_T=0.002 and beta=0.0001; wing strength 0.5 and density strength 0.2. Penalties are normalized to mean diagonal one, then lambda is independently selected. The reference uses a price-fitted SSVI with conservative global sufficient butterfly bounds. All methods share that same reference.')
    p('Pricing uses a clustered 401-node state mesh, quadratic time spacing with quote-expiry knots, and implicit startup damping. Final prices are independently repriced at 801 nodes / 480 nominal time intervals. A wider domain is also checked. The volatility correction amplitude has the same +/-2.5 numerical bounds for all penalties. Final fits reaching the Gauss-Newton budget receive a training-only L-BFGS-B refinement with lambda frozen.')
    p('Euro Stoxx and Bates have no bid-ask fields. A declared vega-based scale (0.005 IV units, price-scale floor 0.00001) is used for fitting; it is not measured market uncertainty. Synthetic Albani observations use their known generation noise scale. Absolute IV errors in nearly zero-price tails are poorly identified; both all-quote and identifiable-quote metrics are retained in the result files.',True)
    new();h('3. Held-out pricing results')
    image('pricing_comparison.png',485)
    p('The no-penalty correction is included to distinguish additional flexibility from regularization. H1 and H2 are conventional penalties applied to the same latent log-variance correction. They are not exact reimplementations of the original papers in volatility coordinates, and these results must not be compared directly with their published in-sample residuals.')
    rows=[['Dataset / protocol','Full vs H1','Full vs H2','Full vs SSVI']]
    for ds in names:
        for proto in protocols:
            vals=[]
            for other in ['h1','h2','ssvi_pde']:
                row=pairs[(pairs.dataset==ds)&(pairs.protocol==proto)&(pairs.comparison=='operator_full vs '+other)].iloc[0]
                d=metrics[(metrics.dataset==ds)&(metrics.protocol==proto)].set_index('method')
                scaled_change=100*(d.loc['operator_full','scaled_price_rmse']/d.loc[other,'scaled_price_rmse']-1)
                vals.append(f'{row.rmse_change_percent:+.1f}% / {scaled_change:+.1f}%')
            rows.append([names[ds]+' / '+protocols[proto],*vals])
    table(rows,[197,96,96,96])
    p('Each cell gives the percentage change in raw price RMSE / scaled price RMSE. Positive means the full operator is worse. The scale is the same vega-based weight used in fitting; method rankings can differ by metric. Identifiable-quote IV errors and valid-IV fractions are also retained in market_and_bates_metrics.csv.',True)
    p('Descriptive maturity-cluster bootstrap intervals for raw price error are saved in paired_comparisons.csv. They do not establish independent-date or market-regime robustness: there is one historical snapshot and one synthetic Bates surface.',True)
    new();h('4. What do the added potentials contribute?')
    p('These ablations retain the H1 derivative and magnitude terms while changing the density and wing potentials. Each shape is trace-normalized and gets its own inner-selected lambda. They isolate the declared configurations rather than keeping a numerically identical shrinkage strength.')
    rows=[['Dataset / protocol','H1','+ density','+ wing','+ both']]
    for ds in names:
        for proto in protocols:
            d=metrics[(metrics.dataset==ds)&(metrics.protocol==proto)].set_index('method')
            rows.append([names[ds]+' / '+protocols[proto],*[f'{d.loc[m,"price_rmse"]*1e4:.3f}' for m in ['h1','h1_density','h1_wing','operator_full']]])
    table(rows,[177,77,77,77,77])
    p('Refined-grid out-of-fold price RMSE, in basis points of spot. Lower is better. Scaled price error and IV errors for every ablation are retained in the result tables.',True)
    sub('Known-truth recovery across all ten noise draws')
    rows=[['Configuration','Full-region mean RMSE','Central mean RMSE']]
    for m,label in [('h1','H1'),('h1_density','H1 + density'),('h1_wing','H1 + wing'),('operator_full','H1 + both')]:
        d=synth[synth.method==m].set_index('region')
        rows.append([label,f'{d.loc["full","mean"]:.5f}',f'{d.loc["central","mean"]:.5f}'])
    table(rows,[161,162,162])
    p('The potentials can help only when their extra shrinkage is appropriate for the actual departure from the reference. A sparse location is a location with less information; it need not be a location with a smaller true correction. Consequently, improved stability alone does not establish improved accuracy.')
    p('These results apply to density strength 0.2, wing strength 0.5, the declared bandwidths and common reference family. The potential strengths themselves were not optimized against outer test outcomes. The study does not establish the best possible member of either penalty family.',True)
    new();h('5. Recovery when the true surface is known')
    image('synthetic_recovery.png',485)
    rows=[['Method','Full mean','Full median','Central mean','Central median']]
    for m in ['ssvi_pde','unregularized','h1','h2','operator_full']:
        d=synth[synth.method==m].set_index('region')
        rows.append([methodnames.get(m,'No penalty'),f'{d.loc["full","mean"]:.5f}',f'{d.loc["full","median"]:.5f}',f'{d.loc["central","mean"]:.5f}',f'{d.loc["central","median"]:.5f}'])
    table(rows,[141,86,86,86,86])
    p('Each dot is one fresh noise draw; horizontal marks are means. The full region is log-moneyness [-0.75,0.75] and maturity [0.1,0.5]; the central region restricts |x| to 0.35. Neither the truth nor the evaluation grid selected lambda. This is ten noise realizations of one truth, not ten independently chosen true surfaces.')
    for _,r in synthpairs.iterrows():
        p(f'{r.region.capitalize()} region: {r.comparison.replace("operator_full","Full operator")} wins {int(r.wins)}/{int(r.draws)} draws; change in mean RMSE {r.mean_rmse_change_percent:+.1f}%.',True)
    p(f'The generation-grid refinement RMSE is {generation["rmse_difference"]:.2g}, versus mean quote-noise SD {generation["price_noise_sd_mean"]:.2g}. Generation and inversion use different grids; both use the verified PDE scheme. The published truth has piecewise boundaries, which continuous smooth corrections cannot reproduce exactly.',True)
    noisy=[r for r in input_checks if r['dataset'].startswith('albani')]
    p(f'The fresh Gaussian price noise, including the 0.000001 SD floor, produces {sum(r["below_intrinsic"] for r in noisy)} of {sum(r["n"] for r in noisy)} observations below the intrinsic-value bound. These noisy observations are retained without clipping; an arbitrage-free model cannot fit them exactly. Surface recovery is evaluated against the clean known truth. Neither supplied Euro Stoxx nor Bates inputs violates this bound.',True)
    new();h('6. What the recovered surface looks like')
    image('albani_surface_slices.png',490)
    p('Predeclared illustration: noise seed 20261022, with three fixed maturity slices. The black curve is the published synthetic truth. Plot selection does not depend on which method wins. Surface arrays for every seed and model are included in the archive.')
    sub('Numerical and optimization diagnostics')
    table([['Check','Recorded result'],['Completed / diagnosed outer folds',f'{numerical["completed_folds"]} / {numerical["diagnosed_folds"]}'],['Convexity violations on tested grids',str(numerical['convexity_violations'])],['Calendar violations on tested grids',str(numerical['calendar_violations'])],['Minimum time value',f'{numerical["minimum_time_value"]:.3g}'],['Largest refined-grid test-price RMSE change',f'{numerical["max_refinement_rmse"]:.3g}'],['Largest widened-domain test-price RMSE change',f'{numerical["max_domain_rmse"]:.3g}'],['Selected fits touching amplitude bounds (before polish)',str(numerical['selected_fits_with_active_bounds'])+' / '+str(numerical['selected_correction_fits'])],['Final fits sent to training-only optimization refinement',str(numerical['training_only_polished_fits'])],['Refinement optimizer success flags',str(numerical['polished_optimizer_success'])]],[326,159])
    p('Convexity tolerance: slope-increment below -0.0000001; calendar tolerance: price increment below -0.000000001. These are finite-grid diagnostics, not a continuum theorem. Widening the domain also changes the clustered spatial mesh slightly. Detailed per-fold solver stopping reasons, bound fractions and refinement histories are retained. Optimizer success is a local numerical stopping criterion, not proof of a global optimum.',True)
    p(f'{numerical["selected_lambdas_at_search_boundary"]} selected lambdas lie at an endpoint of the tested range, including its deterministic extension where used. Their preferred strengths are not fully bracketed by this finite search. Per-method details are in lambda_search_diagnostics.csv.',True)
    sub('Implementation counterexamples')
    p(f'The audit fixture has positive increasing ATM knots, but old cubic extrapolation gives theta derivative {math["pchip_counterexample"]["old_derivative"]:.5f} at T=2. The fixed positive linear tail gives {math["pchip_counterexample"]["fixed_derivative"]:.5f}. The previous constant-function integral on a 6-by-11 grid was 1.32 instead of 1; corrected quadrature returns exactly 1.',True)
    new();h('7. Correction-grid sensitivity and limits')
    sensitivity=ROOT/'outputs/stage_25_parameter_grid_sensitivity'
    rows=[['Dataset / method','105 parameters','171 parameters','Change']]
    for ds in ['eurostoxx_2010','bates_author','albani_noise_00']:
        source=STAGE/ds/('noise_recovery' if ds.startswith('albani') else 'quote_cv')/'fold_00/predictions.csv'
        a=pd.read_csv(source);b=pd.read_csv(sensitivity/ds/'quote_cv/fold_00/predictions.csv')
        for m in ['h1','h2','operator_full']:
            def rmse(d):
                q=d[d.method==m];return float(np.sqrt(np.mean((q.predicted_price-q.observed_price)**2)))
            aa,bb=rmse(a),rmse(b)
            rows.append([names.get(ds,'Albani seed 0')+' / '+methodnames[m],f'{aa*1e4:.3f}',f'{bb*1e4:.3f}',f'{100*(bb/aa-1):+.1f}%'])
    table(rows,[216,91,91,87])
    p('These are coarse-pricer, fixed-budget results for predeclared outer fold 0 only. Lambda is retuned on inner data for the denser grid. This sensitivity was not used to choose a preferred reported model. It is not a second independent benchmark.',True)
    sub('Scientific limits')
    for text in [
        'The result concerns this parameterization, potential strengths, shared reference family, finite optimization budget and datasets. It does not establish that any regularizer universally dominates another.',
        'The conservative SSVI constraints can limit baseline flexibility. Improvement over that reference is not proof of improvement over every possible SSVI fit; the comparison between penalties is better controlled.',
        'Inner validation holds out strikes for both protocols. It is aligned with quote interpolation, but can select a poor lambda for whole-expiry omission or maturity extrapolation. The expiry results stress this declared selection procedure rather than an optimally tuned maturity-prediction procedure.',
        'Market and Bates inputs do not reveal a true local-volatility surface. Better quote fit need not mean better hedging or better recovery in unquoted regions.',
        'The authors neural-network model, original Geng volatility-space optimizer and original Albani noisy realization were not reproduced. The data are reused for a controlled penalty comparison.',
        'Gaussian-process, historical Laplace, trading and regime-forecasting extensions are outside this audit. No Sharpe ratio or profitability inference is made.']:
        p('- '+text,True)
    new();h('8. Can superiority be proved without data?')
    p('<b>Conditional guarantees are possible; universal superiority does not follow.</b> Better numerical stability and lower estimation error are different claims. The complete derivations and assumptions are in reports/Regularizer_Theory.md.')
    sub('An exact comparison in the fixed linearized problem')
    p('Freeze the reference, weights and operator. Write standardized observations as b = A h* + noise, with zero-mean noise covariance Sigma. For tau = n lambda / p and G = A^T A + tau R, the unconstrained estimate is h_hat = K b, where K = G^(-1) A^T.')
    table([['Target','Exact mean squared error'],['Any fixed linear map B','|| B(KA-I)h* ||^2 + trace(B K Sigma K^T B^T)'],['Latent amplitude','B = I'],['Actual log-variance correction','B = taper times interpolation'],['Held-out prices','Use their Jacobian for a first-order approximation only']],[163,322])
    p('A stronger potential can reduce variance while increasing bias toward the SSVI reference. Comparing this expression for the two operators gives a sufficient and necessary risk comparison in this fixed linear model. It does not by itself cover fitted references, selected lambdas, box constraints or nonlinear PDE effects.')
    sub('A counterexample to unconditional improvement')
    p('For b = h* + noise with noise variance 1, a ridge strength t gives risk (t^2 h*^2 + 1)/(1+t)^2. Increasing t from 1 to 2 is better when h*^2 &lt; 5/7, and worse when h*^2 &gt; 5/7. Extra shrinkage cannot be assumed to improve every possible truth. Trace normalization and separate lambda tuning also mean the fitted full operator is not simply a larger matrix than fitted H1.')
    sub('What would support a useful positive theorem?')
    p('In a linear Gaussian experiment with known prior h* ~ N(0,C) and independent noise variance s^2, choosing tau R = s^2 C^(-1) gives the posterior mean, which minimizes prior-averaged squared error. This justifies the proposed operator only if its precision describes the actual correction distribution. Quote sparsity alone does not prove that assumption.')
    p('Positive beta and nonnegative potentials give a positive-definite discrete penalty and a unique regularized linearized solution. A continuous finite-dimensional objective on the implemented compact parameter box attains a minimum. Neither result gives uniqueness of the nonlinear solution or success of a local optimizer. A theorem comparing rates would need an explicit truth class, target norm, noise model, numerical error bound and regularization schedule.')
    p('For a fixed bounded nonnegative potential, the energy obeys E_H1 &lt;= E_full &lt;= (1 + Vmax/beta) E_H1 before trace normalization. Thus the added potential gives an equivalent H1 norm, changing spatial shrinkage rather than the function-space smoothness order. That fact alone does not yield a better convergence rate.',True)
    p('Relevant existing theory includes Albani, De Cezaro and Zubelli (arXiv:1308.2659), on discrete regularization and convergence rates, and Albani and Zubelli (arXiv:1211.0170), on online calibration with Morozov selection. Their results provide a starting point; extending them to this tapered log-variance model requires checking their hypotheses.',True)
    new();h('9. Reproducibility and research interpretation')
    p('The useful empirical question is whether spatial shrinkage adds value beyond a conventional smoothness penalty. The supplied files let that question be checked at the level of individual held-out quotes, complete expiries, known-truth surfaces and optimizer histories. The results do not establish novelty by absence of a matching paper.')
    p('The closest mathematical precedent is reference-centered H1/Tikhonov calibration, including Crepey (2003). Ackerer et al. supply a related classical-model-plus-correction architecture in implied-volatility coordinates. reports/Literature_Comparison.md records the overlap, differences, data access and limits of this comparison, including two more recent adjacent papers.',True)
    sub('Evidence map')
    table([['Path','Contents'],['outputs/implementation_audit.json','Source archive fingerprint; implementation findings and fixes.'],['outputs/stage_23_literature_audit_final/','50 outer folds; train/test indices, inner tuning curves, fitted parameters, predictions and numerical diagnostics.'],['outputs/stage_25_parameter_grid_sensitivity/','Three predeclared fold-0 refits on a denser correction grid.'],['reports/literature_audit/','Pooled and paired comparisons, surface-recovery results, figures and numerical summary.'],['reports/Regularizer_Theory.md','Risk derivation, counterexample, stability and conditional optimality.'],['data/literature/benchmark_inputs/','Retrieved inputs and source fingerprints.'],['tests/test_literature_audit.py','Extrapolation, quadrature, pricing convergence, tangent derivative and outer-value leakage tests.']],[252,233])
    sub('Run order')
    p('Run scripts 23_literature_benchmarks.py, 24_literature_diagnostics.py, 25_parameter_grid_sensitivity.py, 26_analyze_literature_results.py and 27_write_audit_report.py. Commands are in README.md. The outer folds are deterministic; synthetic seeds are 20261022 through 20261031. BLAS threads are fixed to one per worker. Earlier result folders are historical evidence, not regenerated results under the corrected code.',True)
    p('Environment: '+', '.join(f'{k} {v}' for k,v in environment.items()),True)
    sub('Primary literature and data sources')
    for label,url in [
        ('Crepey (2003), local-volatility calibration using Tikhonov regularization','https://doi.org/10.1137/S0036141001400202'),
        ('Geng, Navon and Chen (2014), second-order Tikhonov local-volatility calibration','https://doi.org/10.1080/14697688.2013.819988'),
        ('Albani, Ascher, Yang and Zubelli (2017), Data driven recovery of local volatility surfaces','https://arxiv.org/abs/1512.07660'),
        ('Ackerer, Tagasovska and Vatter (2020), Deep Smoothing of the Implied Volatility Surface','https://arxiv.org/abs/1906.05065'),
        ('QuantLib public benchmark transcription','https://github.com/lballabio/QuantLib/blob/master/test-suite/andreasenhugevolatilityinterpl.cpp'),
        ('Gatheral and Jacquier, Arbitrage-free SVI volatility surfaces','https://arxiv.org/abs/1204.0646')]:
        p(html.escape(label)+':<br/><link href="'+html.escape(url)+'" color="#256286">'+html.escape(url)+'</link>',True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#b8c5d0'));canvas.line(43,39,A4[0]-43,39)
        canvas.setFont('DV',8);canvas.setFillColor(colors.HexColor('#586a7b'))
        canvas.drawString(43,25,'DUPIRE | IMPLEMENTATION AUDIT AND CONTROLLED BENCHMARKS')
        canvas.drawRightString(A4[0]-43,25,str(doc.page))
    doc=SimpleDocTemplate(str(TARGET),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=43,bottomMargin=54,
                         title='Dupire Literature Audit and Benchmarks',author='Fakhruddin Hakim Hussain')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(TARGET)

if __name__=='__main__':main()
