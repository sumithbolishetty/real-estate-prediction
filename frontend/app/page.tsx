"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Options = { locations: string[]; property_types: string[]; parking_options: string[]; amenities: string[]; model_name: string };
type Result = { estimated_price: number; formatted_price: string };
const API = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8100").replace(/\/$/, "");
const empty: Options = { locations: [], property_types: [], parking_options: [], amenities: [], model_name: "Regression model" };

function title(value: string) { return value.replace(/\b\w/g, (letter) => letter.toUpperCase()); }

export default function Home() {
  const [options, setOptions] = useState<Options>(empty);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [optionsError, setOptionsError] = useState("");
  const [requestError, setRequestError] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [form, setForm] = useState({ location: "", area: "", bedrooms: "2", bathrooms: "1", property_type: "", parking: "", property_age: "0", amenities: [] as string[] });

  useEffect(() => {
    fetch(`${API}/api/options`).then(async (response) => {
      if (!response.ok) throw new Error("Prediction options are unavailable. Start the backend and try again.");
      const data = await response.json() as Options;
      if (!Array.isArray(data.locations) || !Array.isArray(data.property_types) ||
          !Array.isArray(data.parking_options) || !Array.isArray(data.amenities)) {
        throw new Error("The backend returned invalid property options.");
      }
      setOptions(data);
      setForm((previous) => ({ ...previous, location: data.locations[0] || "", property_type: data.property_types[0] || "", parking: data.parking_options[0] || "" }));
      if (!data.locations.length) setOptionsError("No locations available");
    }).catch(() => setOptionsError("Unable to load property options. Please make sure the backend is running."))
      .finally(() => setLoadingOptions(false));
  }, []);

  const currency = useMemo(() => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }), []);
  function update(name: string, value: string) { setForm((previous) => ({ ...previous, [name]: value })); }
  function toggleAmenity(value: string) {
    setForm((previous) => ({ ...previous, amenities: previous.amenities.includes(value) ? previous.amenities.filter((item) => item !== value) : [...previous.amenities, value] }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setRequestError(""); setResult(null);
    const area = Number(form.area), bedrooms = Number(form.bedrooms), bathrooms = Number(form.bathrooms), age = Number(form.property_age);
    if (!form.location || !form.property_type || !form.parking) return setRequestError("Choose a location, property type, and parking option.");
    if (!Number.isFinite(area) || area <= 0) return setRequestError("Area must be greater than zero.");
    if (!Number.isInteger(bedrooms) || bedrooms < 1) return setRequestError("Bedrooms must be at least one.");
    if (!Number.isFinite(bathrooms) || bathrooms <= 0) return setRequestError("Bathrooms must be greater than zero.");
    if (!Number.isInteger(age) || age < 0) return setRequestError("Property age cannot be negative.");
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ location: form.location, area, bedrooms, bathrooms, property_type: form.property_type, parking: form.parking, property_age: age, amenities: form.amenities }) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The prediction request was rejected. Check the form and try again.");
      if (typeof data.estimated_price !== "number") throw new Error("The prediction service returned an invalid response.");
      setResult(data as Result);
    } catch (error) {
      setRequestError(error instanceof TypeError
        ? "Unable to reach the prediction service. Please make sure the backend is running."
        : error instanceof Error ? error.message : "Prediction failed. Please try again.");
    }
    finally { setBusy(false); }
  }

  return <main className="page-shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-icon">⌂</span><span>Real Estate Price Prediction System</span></a><span className="status"><i /> Machine learning estimate</span></header>
    <section className="hero"><div className="hero-copy"><div className="eyebrow"><span /> AI-POWERED PROPERTY PRICE ESTIMATION</div><h1>Find your property’s<br /><em>estimated value.</em></h1><p>Share a few details about the home. Our model will provide a data-driven price estimate in seconds.</p><div className="hero-note"><span>✦</span> A helpful starting point for your next move</div></div><div className="hero-art" aria-hidden="true"><div className="sun"/><div className="building building-back"><div/><div/><div/><div/></div><div className="building building-front"><div/><div/><div/><div/><div/><div/></div><div className="ground"/><div className="art-label">PROPERTY INSIGHTS <span>01 / 03</span></div></div></section>
    <div className="content-grid"><section className="form-card"><div className="section-heading"><div><div className="eyebrow small">PROPERTY DETAILS</div><h2>Tell us about the home</h2></div><span className="step-label">01 <span>/ 01</span></span></div>
      {loadingOptions && <div className="notice">Connecting to the prediction service…</div>}
      {optionsError && <div role="alert" className="error-box"><span>!</span>{optionsError}</div>}
      <form onSubmit={submit}>
        <div className="field-grid">
          <label className="field wide"><span>Location</span><span className="select-wrap"><select required value={form.location} onChange={(e) => update("location", e.target.value)} disabled={loadingOptions || !options.locations.length}><option value="">{optionsError === "No locations available" ? "No locations available" : "Choose a city"}</option>{options.locations.map((value) => <option key={value} value={value}>{title(value)}</option>)}</select><b>⌄</b></span></label>
          <label className="field"><span>Area <small>(sq ft)</small></span><input type="number" min="1" step="any" placeholder="e.g. 1,200" required value={form.area} onChange={(e) => update("area", e.target.value)} /></label>
          <label className="field"><span>Property age <small>(years)</small></span><input type="number" min="0" step="1" required value={form.property_age} onChange={(e) => update("property_age", e.target.value)} /></label>
          <label className="field"><span>Bedrooms</span><input type="number" min="1" step="1" required value={form.bedrooms} onChange={(e) => update("bedrooms", e.target.value)} /></label>
          <label className="field"><span>Bathrooms</span><input type="number" min="0.1" step="any" required value={form.bathrooms} onChange={(e) => update("bathrooms", e.target.value)} /></label>
          <label className="field"><span>Property type</span><span className="select-wrap"><select required value={form.property_type} onChange={(e) => update("property_type", e.target.value)} disabled={loadingOptions || !options.property_types.length}><option value="">{options.property_types.length ? "Choose type" : "No property types available"}</option>{options.property_types.map((value) => <option key={value} value={value}>{title(value)}</option>)}</select><b>⌄</b></span></label>
          <label className="field"><span>Parking</span><span className="select-wrap"><select required value={form.parking} onChange={(e) => update("parking", e.target.value)} disabled={loadingOptions || !options.parking_options.length}><option value="">{options.parking_options.length ? "Choose option" : "No parking options available"}</option>{options.parking_options.map((value) => <option key={value} value={value}>{title(value)}</option>)}</select><b>⌄</b></span></label>
        </div>
        <fieldset className="amenities"><legend>Amenities <small>(optional)</small></legend><div className="chips">{options.amenities.map((amenity) => <button className={`chip ${form.amenities.includes(amenity) ? "selected" : ""}`} type="button" key={amenity} aria-pressed={form.amenities.includes(amenity)} onClick={() => toggleAmenity(amenity)}><span>{form.amenities.includes(amenity) ? "✓" : "+"}</span>{amenity}</button>)}</div></fieldset>
        {requestError && <div role="alert" className="error-box"><span>!</span>{requestError}</div>}
        <button className="submit-button" type="submit" disabled={busy || loadingOptions || Boolean(optionsError) || !options.locations.length || !options.property_types.length || !options.parking_options.length}>{busy ? <><i className="spinner"/> Predicting…</> : <>Predict Property Price <span>→</span></>}</button>
      </form>
    </section>
    <aside className="side-column">{result ? <section className="result-card"><div className="result-icon">₹</div><div className="eyebrow small">YOUR ESTIMATE</div><h2>Estimated Property<br/>Price</h2><div className="price">₹{currency.format(result.estimated_price)}</div><p>Generated from the details you provided.</p><div className="result-footer"><span>✦</span> Machine learning estimate</div></section> : <section className="result-placeholder"><div className="placeholder-orbit orbit-one"/><div className="placeholder-orbit orbit-two"/><div className="placeholder-home">⌂</div><div className="eyebrow small">YOUR ESTIMATE</div><h3>Your estimate<br/>will appear here</h3><p>Complete the property details and we’ll calculate an estimated value.</p></section>}
      <section className="model-card"><div className="model-mark">✳</div><div><strong>Built for a quick estimate</strong><p>{options.model_name} · 8 feature groups</p></div></section>
      <p className="disclaimer">Estimated price generated by the machine-learning model. Actual market prices may vary based on factors not included in the dataset.</p>
    </aside></div>
    <footer><span>PROPERTY ESTIMATE <i>·</i> DEMO</span><span>Designed to help you explore possibilities.</span></footer>
  </main>;
}
